"""MoE routing pattern analysis for expert load balancing and optimization.

Analyzes expert selection patterns across tokens and layers to answer:
1. How balanced are expert loads in practice?
2. Do certain experts get selected together frequently?
3. Can we predict routing from early layers?

These insights enable:
- Pre-loading frequently used experts
- Batching tokens going to same expert
- Expert pruning (removing never-used experts)

Usage:
    from metal_marlin.analysis import MoERoutingProfiler

    # Create profiler
    profiler = MoERoutingProfiler(num_experts=64, num_layers=40, top_k=2)

    # During inference, record routing decisions
    for layer_idx in range(num_layers):
        expert_ids, expert_probs = router(hidden_states)
        profiler.record_routing(layer_idx, expert_ids, expert_probs)

    # Analyze patterns
    report = profiler.generate_report()
    profiler.plot_routing_heatmap("routing_analysis.png")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExpertLoadStats:
    """Load statistics for a single expert across all tokens.

    Attributes:
        expert_id: Expert index (0 to num_experts-1)
        total_activations: Total number of tokens routed to this expert
        activation_rate: Fraction of total routing decisions selecting this expert
        avg_probability: Average routing probability when selected
        std_probability: Standard deviation of routing probability
        rank_distribution: How often selected as 1st, 2nd, etc. choice
        is_hot: True if expert is frequently used (>1.5x average load)
        is_cold: True if expert is rarely used (<0.5x average load)
        is_dead: True if expert was never selected
    """

    expert_id: int
    total_activations: int = 0
    activation_rate: float = 0.0
    avg_probability: float = 0.0
    std_probability: float = 0.0
    rank_distribution: dict[int, int] = field(default_factory=dict)
    is_hot: bool = False
    is_cold: bool = False
    is_dead: bool = False

    @property
    def primary_selection_rate(self) -> float:
        """Fraction of activations where this expert was top-1 choice."""
        if self.total_activations == 0:
            return 0.0
        return self.rank_distribution.get(0, 0) / self.total_activations


@dataclass
class ExpertCooccurrence:
    """Co-occurrence statistics between expert pairs.

    Tracks how often pairs of experts are selected together for the same token,
    which informs batching strategies and cache prefetching.

    Attributes:
        cooccurrence_matrix: [num_experts, num_experts] symmetric matrix
            where entry (i, j) counts how often experts i and j were both
            selected for the same token.
        conditional_probs: P(expert_j | expert_i) - probability of selecting j
            given that i was selected.
        top_pairs: List of (expert_i, expert_j, count) sorted by frequency.
    """

    num_experts: int
    cooccurrence_matrix: np.ndarray = field(default=None)
    conditional_probs: np.ndarray | None = None
    top_pairs: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cooccurrence_matrix is None:
            self.cooccurrence_matrix = np.zeros(
                (self.num_experts, self.num_experts), dtype=np.int64
            )


@dataclass
class LayerRoutingProfile:
    """Routing profile for a single transformer layer.

    Attributes:
        layer_idx: Layer index (0 to num_layers-1)
        expert_stats: Per-expert statistics for this layer
        total_tokens: Total tokens processed through this layer
        load_balance_cv: Coefficient of variation of expert loads (lower = more balanced)
        entropy: Routing entropy (higher = more uniform distribution)
        gini_coefficient: Gini coefficient of expert loads (0 = perfect equality)
        active_experts: Number of experts that received at least one token
        dead_experts: Number of experts that received no tokens
    """

    layer_idx: int
    expert_stats: dict[int, ExpertLoadStats] = field(default_factory=dict)
    total_tokens: int = 0
    load_balance_cv: float = 0.0
    entropy: float = 0.0
    gini_coefficient: float = 0.0
    active_experts: int = 0
    dead_experts: int = 0

    def get_hot_experts(self, threshold: float = 1.5) -> list[int]:
        """Get experts with activation rate > threshold * average."""
        return [
            eid for eid, stats in self.expert_stats.items()
            if stats.is_hot
        ]

    def get_cold_experts(self, threshold: float = 0.5) -> list[int]:
        """Get experts with activation rate < threshold * average."""
        return [
            eid for eid, stats in self.expert_stats.items()
            if stats.is_cold
        ]

    def get_dead_experts(self) -> list[int]:
        """Get experts that were never selected."""
        return [
            eid for eid, stats in self.expert_stats.items()
            if stats.is_dead
        ]


@dataclass
class RoutingPredictability:
    """Analysis of how predictable later-layer routing is from early layers.

    If routing is predictable, we can prefetch experts before they're needed.

    Attributes:
        layer_correlations: [num_layers, num_layers] correlation matrix
            where entry (i, j) is the correlation between expert selections
            at layers i and j.
        predictable_from_layer: Earliest layer from which routing is predictable
            (correlation > threshold with later layers).
        avg_prediction_accuracy: Average accuracy of predicting later-layer
            routing from the predictable_from_layer.
    """

    num_layers: int
    layer_correlations: np.ndarray = field(default=None)
    predictable_from_layer: int = -1
    avg_prediction_accuracy: float = 0.0
    per_layer_accuracy: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_correlations is None:
            self.layer_correlations = np.zeros(
                (self.num_layers, self.num_layers), dtype=np.float32
            )


class MoERoutingProfiler:
    """Profile MoE routing patterns for optimization insights.

    Collects routing decisions across tokens and layers to analyze:
    - Expert load balance
    - Expert co-occurrence patterns
    - Layer-to-layer routing predictability
    - Hot/cold/dead expert identification

    Thread-safe for concurrent recording during batch inference.

    Args:
        num_experts: Total number of experts per MoE layer
        num_layers: Number of MoE layers in the model
        top_k: Number of experts selected per token
        hot_threshold: Activation rate multiplier above average to be "hot"
        cold_threshold: Activation rate multiplier below average to be "cold"
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        top_k: int = 2,
        hot_threshold: float = 1.5,
        cold_threshold: float = 0.5,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.top_k = top_k
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold

        # Per-layer data collection
        # Shape: [num_layers, num_samples, top_k] for expert_ids
        self._expert_ids: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
        self._expert_probs: list[list[np.ndarray]] = [[] for _ in range(num_layers)]

        # Computed results (lazily populated)
        self._layer_profiles: dict[int, LayerRoutingProfile] | None = None
        self._cooccurrence: ExpertCooccurrence | None = None
        self._predictability: RoutingPredictability | None = None

    def record_routing(
        self,
        layer_idx: int,
        expert_ids: np.ndarray,
        expert_probs: np.ndarray | None = None,
    ) -> None:
        """Record routing decisions for a batch of tokens.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            expert_ids: [batch_size, top_k] or [batch_size * seq_len, top_k]
                Expert indices selected for each token.
            expert_probs: Optional [batch_size, top_k] routing probabilities.
                If not provided, uniform weights are assumed.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

        # Convert to numpy if needed
        if hasattr(expert_ids, "numpy"):
            expert_ids = expert_ids.numpy()
        elif hasattr(expert_ids, "tolist"):
            expert_ids = np.array(expert_ids)

        if expert_probs is not None:
            if hasattr(expert_probs, "numpy"):
                expert_probs = expert_probs.numpy()
            elif hasattr(expert_probs, "tolist"):
                expert_probs = np.array(expert_probs)

        # Ensure 2D: [tokens, top_k]
        if expert_ids.ndim == 1:
            expert_ids = expert_ids.reshape(-1, self.top_k)

        # Store data
        self._expert_ids[layer_idx].append(expert_ids.astype(np.int32))
        if expert_probs is not None:
            self._expert_probs[layer_idx].append(expert_probs.astype(np.float32))

        # Invalidate cached results
        self._layer_profiles = None
        self._cooccurrence = None
        self._predictability = None

    def _compute_layer_profiles(self) -> dict[int, LayerRoutingProfile]:
        """Compute per-layer routing statistics."""
        profiles = {}

        for layer_idx in range(self.num_layers):
            if not self._expert_ids[layer_idx]:
                profiles[layer_idx] = LayerRoutingProfile(layer_idx=layer_idx)
                continue

            # Concatenate all recorded batches
            all_ids = np.concatenate(self._expert_ids[layer_idx], axis=0)
            total_tokens = all_ids.shape[0]

            all_probs = None
            if self._expert_probs[layer_idx]:
                all_probs = np.concatenate(self._expert_probs[layer_idx], axis=0)

            # Count activations per expert
            expert_counts = np.zeros(self.num_experts, dtype=np.int64)
            rank_counts: dict[int, dict[int, int]] = {
                e: {k: 0 for k in range(self.top_k)}
                for e in range(self.num_experts)
            }

            # Sum probabilities per expert for averaging
            prob_sums = np.zeros(self.num_experts, dtype=np.float64)
            prob_sq_sums = np.zeros(self.num_experts, dtype=np.float64)

            for token_idx in range(total_tokens):
                for k in range(self.top_k):
                    expert_id = all_ids[token_idx, k]
                    expert_counts[expert_id] += 1
                    rank_counts[expert_id][k] += 1

                    if all_probs is not None:
                        prob = all_probs[token_idx, k]
                        prob_sums[expert_id] += prob
                        prob_sq_sums[expert_id] += prob * prob

            # Compute per-expert statistics
            total_assignments = total_tokens * self.top_k
            avg_activations = total_assignments / self.num_experts

            expert_stats = {}
            active_count = 0
            dead_count = 0

            for expert_id in range(self.num_experts):
                count = expert_counts[expert_id]
                rate = count / total_assignments if total_assignments > 0 else 0.0

                avg_prob = prob_sums[expert_id] / count if count > 0 else 0.0
                var_prob = (
                    (prob_sq_sums[expert_id] / count - avg_prob ** 2)
                    if count > 0 else 0.0
                )
                std_prob = np.sqrt(max(0, var_prob))

                is_hot = count > self.hot_threshold * avg_activations
                is_cold = count < self.cold_threshold * avg_activations and count > 0
                is_dead = count == 0

                if is_dead:
                    dead_count += 1
                else:
                    active_count += 1

                expert_stats[expert_id] = ExpertLoadStats(
                    expert_id=expert_id,
                    total_activations=int(count),
                    activation_rate=rate,
                    avg_probability=avg_prob,
                    std_probability=std_prob,
                    rank_distribution=rank_counts[expert_id].copy(),
                    is_hot=is_hot,
                    is_cold=is_cold,
                    is_dead=is_dead,
                )

            # Compute load balance metrics
            counts_nonzero = expert_counts[expert_counts > 0]
            cv = (
                np.std(counts_nonzero) / np.mean(counts_nonzero)
                if len(counts_nonzero) > 0 and np.mean(counts_nonzero) > 0
                else 0.0
            )

            # Routing entropy (using softmax of counts as distribution)
            probs_norm = expert_counts / expert_counts.sum() if expert_counts.sum() > 0 else np.zeros_like(expert_counts)
            probs_norm = probs_norm[probs_norm > 0]  # Remove zeros for log
            entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-10))
            max_entropy = np.log(self.num_experts)  # Uniform distribution
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Gini coefficient
            sorted_counts = np.sort(expert_counts)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            gini = (
                (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                if n > 0 and cumsum[-1] > 0 else 0.0
            )

            profiles[layer_idx] = LayerRoutingProfile(
                layer_idx=layer_idx,
                expert_stats=expert_stats,
                total_tokens=total_tokens,
                load_balance_cv=float(cv),
                entropy=float(normalized_entropy),
                gini_coefficient=float(gini),
                active_experts=active_count,
                dead_experts=dead_count,
            )

        return profiles

    def _compute_cooccurrence(self) -> ExpertCooccurrence:
        """Compute expert co-occurrence matrix across all layers."""
        cooccurrence = ExpertCooccurrence(num_experts=self.num_experts)

        # Aggregate co-occurrences across all layers
        for layer_idx in range(self.num_layers):
            if not self._expert_ids[layer_idx]:
                continue

            all_ids = np.concatenate(self._expert_ids[layer_idx], axis=0)

            for token_idx in range(all_ids.shape[0]):
                experts = all_ids[token_idx]
                # Count all pairs (including self-pairs on diagonal)
                for i in range(self.top_k):
                    for j in range(i, self.top_k):
                        e1, e2 = int(experts[i]), int(experts[j])
                        cooccurrence.cooccurrence_matrix[e1, e2] += 1
                        if e1 != e2:
                            cooccurrence.cooccurrence_matrix[e2, e1] += 1

        # Compute conditional probabilities P(j | i)
        row_sums = cooccurrence.cooccurrence_matrix.sum(axis=1, keepdims=True)
        cooccurrence.conditional_probs = np.divide(
            cooccurrence.cooccurrence_matrix.astype(np.float32),
            row_sums,
            out=np.zeros_like(cooccurrence.cooccurrence_matrix, dtype=np.float32),
            where=row_sums > 0,
        )

        # Find top co-occurring pairs (exclude diagonal)
        matrix = cooccurrence.cooccurrence_matrix.copy()
        np.fill_diagonal(matrix, 0)  # Exclude self-pairs

        # Get top pairs
        flat_indices = np.argsort(matrix.ravel())[::-1]
        top_pairs = []
        seen_pairs = set()

        for idx in flat_indices[:100]:  # Top 100 pairs
            i, j = divmod(idx, self.num_experts)
            if i > j:  # Only count each pair once
                i, j = j, i
            if (i, j) not in seen_pairs and matrix[i, j] > 0:
                seen_pairs.add((i, j))
                top_pairs.append((i, j, int(matrix[i, j])))

        cooccurrence.top_pairs = top_pairs[:50]  # Keep top 50

        return cooccurrence

    def _compute_predictability(self) -> RoutingPredictability:
        """Analyze how predictable routing is from early layers.

        Computes correlation between expert selections at different layers
        to determine if early-layer routing predicts later-layer routing.
        """
        predictability = RoutingPredictability(num_layers=self.num_layers)

        # Build expert selection vectors per layer
        # For each token, create a binary vector of which experts were selected
        layer_selections: list[np.ndarray | None] = [None] * self.num_layers

        for layer_idx in range(self.num_layers):
            if not self._expert_ids[layer_idx]:
                continue

            all_ids = np.concatenate(self._expert_ids[layer_idx], axis=0)
            num_tokens = all_ids.shape[0]

            # Create binary selection matrix [num_tokens, num_experts]
            selections = np.zeros((num_tokens, self.num_experts), dtype=np.float32)
            for token_idx in range(num_tokens):
                for k in range(self.top_k):
                    selections[token_idx, all_ids[token_idx, k]] = 1.0

            layer_selections[layer_idx] = selections

        # Compute layer-to-layer correlations
        for i in range(self.num_layers):
            if layer_selections[i] is None:
                continue
            for j in range(i, self.num_layers):
                if layer_selections[j] is None:
                    continue

                # Check if same number of tokens (might differ if recording was partial)
                n_i = layer_selections[i].shape[0]
                n_j = layer_selections[j].shape[0]
                n = min(n_i, n_j)

                if n == 0:
                    continue

                # Flatten and compute correlation
                sel_i = layer_selections[i][:n].ravel()
                sel_j = layer_selections[j][:n].ravel()

                # Pearson correlation
                mean_i = np.mean(sel_i)
                mean_j = np.mean(sel_j)
                cov = np.mean((sel_i - mean_i) * (sel_j - mean_j))
                std_i = np.std(sel_i)
                std_j = np.std(sel_j)

                if std_i > 0 and std_j > 0:
                    corr = cov / (std_i * std_j)
                else:
                    corr = 1.0 if i == j else 0.0

                predictability.layer_correlations[i, j] = corr
                predictability.layer_correlations[j, i] = corr

        # Find earliest predictable layer
        # A layer is "predictable from" if correlation with all later layers > 0.7
        threshold = 0.7
        for i in range(self.num_layers):
            if layer_selections[i] is None:
                continue

            # Check correlation with all later layers
            later_corrs = predictability.layer_correlations[i, i + 1 :]
            if len(later_corrs) > 0 and np.all(later_corrs > threshold):
                predictability.predictable_from_layer = i
                predictability.avg_prediction_accuracy = float(np.mean(later_corrs))
                break

        # Per-layer prediction accuracy (from layer 0)
        for j in range(self.num_layers):
            predictability.per_layer_accuracy[j] = float(
                predictability.layer_correlations[0, j]
            )

        return predictability

    @property
    def layer_profiles(self) -> dict[int, LayerRoutingProfile]:
        """Get per-layer routing profiles (computed on first access)."""
        if self._layer_profiles is None:
            self._layer_profiles = self._compute_layer_profiles()
        return self._layer_profiles

    @property
    def cooccurrence(self) -> ExpertCooccurrence:
        """Get expert co-occurrence analysis (computed on first access)."""
        if self._cooccurrence is None:
            self._cooccurrence = self._compute_cooccurrence()
        return self._cooccurrence

    @property
    def predictability(self) -> RoutingPredictability:
        """Get routing predictability analysis (computed on first access)."""
        if self._predictability is None:
            self._predictability = self._compute_predictability()
        return self._predictability

    def get_hot_experts(self, layer_idx: int | None = None) -> list[int]:
        """Get list of hot experts (high load).

        Args:
            layer_idx: If provided, get hot experts for specific layer.
                If None, get experts that are hot in any layer.
        """
        if layer_idx is not None:
            return self.layer_profiles[layer_idx].get_hot_experts()

        hot = set()
        for profile in self.layer_profiles.values():
            hot.update(profile.get_hot_experts())
        return sorted(hot)

    def get_cold_experts(self, layer_idx: int | None = None) -> list[int]:
        """Get list of cold experts (low load).

        Args:
            layer_idx: If provided, get cold experts for specific layer.
                If None, get experts that are cold in any layer.
        """
        if layer_idx is not None:
            return self.layer_profiles[layer_idx].get_cold_experts()

        cold = set()
        for profile in self.layer_profiles.values():
            cold.update(profile.get_cold_experts())
        return sorted(cold)

    def get_dead_experts(self, layer_idx: int | None = None) -> list[int]:
        """Get list of dead experts (never selected).

        Args:
            layer_idx: If provided, get dead experts for specific layer.
                If None, get experts that are dead in ALL layers (prunable).
        """
        if layer_idx is not None:
            return self.layer_profiles[layer_idx].get_dead_experts()

        # Experts dead in all layers are candidates for pruning
        dead_all_layers = set(range(self.num_experts))
        for profile in self.layer_profiles.values():
            if profile.total_tokens > 0:  # Only consider layers with data
                dead_this_layer = set(profile.get_dead_experts())
                dead_all_layers &= dead_this_layer
        return sorted(dead_all_layers)

    def get_prefetch_recommendations(
        self,
        layer_idx: int,
        num_experts: int = 4,
    ) -> list[int]:
        """Get recommended experts to prefetch for a layer.

        Based on historical activation patterns, returns experts most likely
        to be needed.

        Args:
            layer_idx: Target layer to prefetch for
            num_experts: Number of experts to recommend

        Returns:
            List of expert IDs sorted by likelihood of activation
        """
        profile = self.layer_profiles[layer_idx]

        # Sort experts by activation rate
        sorted_experts = sorted(
            profile.expert_stats.values(),
            key=lambda s: s.activation_rate,
            reverse=True,
        )

        return [s.expert_id for s in sorted_experts[:num_experts]]

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive routing analysis report.

        Returns:
            Dictionary containing all analysis results suitable for JSON export.
        """
        # Force computation
        _ = self.layer_profiles
        _ = self.cooccurrence
        _ = self.predictability

        # Aggregate statistics
        total_tokens = sum(p.total_tokens for p in self.layer_profiles.values())
        avg_cv = np.mean([p.load_balance_cv for p in self.layer_profiles.values() if p.total_tokens > 0])
        avg_entropy = np.mean([p.entropy for p in self.layer_profiles.values() if p.total_tokens > 0])
        avg_gini = np.mean([p.gini_coefficient for p in self.layer_profiles.values() if p.total_tokens > 0])

        # Per-layer summary
        layer_summaries = []
        for layer_idx, profile in self.layer_profiles.items():
            if profile.total_tokens == 0:
                continue

            layer_summaries.append({
                "layer": layer_idx,
                "tokens": profile.total_tokens,
                "active_experts": profile.active_experts,
                "dead_experts": profile.dead_experts,
                "load_balance_cv": round(profile.load_balance_cv, 4),
                "entropy": round(profile.entropy, 4),
                "gini": round(profile.gini_coefficient, 4),
                "hot_experts": profile.get_hot_experts(),
                "cold_experts": profile.get_cold_experts(),
            })

        # Global hot/cold/dead experts
        global_hot = self.get_hot_experts()
        global_cold = self.get_cold_experts()
        global_dead = self.get_dead_experts()

        # Top co-occurring pairs (convert numpy int64 to Python int)
        top_pairs = [
            {"experts": [int(p[0]), int(p[1])], "count": int(p[2])}
            for p in self.cooccurrence.top_pairs[:20]
        ]

        return {
            "summary": {
                "num_experts": self.num_experts,
                "num_layers": self.num_layers,
                "top_k": self.top_k,
                "total_tokens_profiled": total_tokens,
                "avg_load_balance_cv": round(float(avg_cv), 4) if not np.isnan(avg_cv) else None,
                "avg_entropy": round(float(avg_entropy), 4) if not np.isnan(avg_entropy) else None,
                "avg_gini": round(float(avg_gini), 4) if not np.isnan(avg_gini) else None,
            },
            "load_balance": {
                "hot_experts": global_hot,
                "cold_experts": global_cold,
                "dead_experts": global_dead,
                "num_hot": len(global_hot),
                "num_cold": len(global_cold),
                "num_dead": len(global_dead),
                "prunable_experts": global_dead,  # Same as dead
            },
            "cooccurrence": {
                "top_pairs": top_pairs,
            },
            "predictability": {
                "predictable_from_layer": self.predictability.predictable_from_layer,
                "avg_prediction_accuracy": round(
                    self.predictability.avg_prediction_accuracy, 4
                ),
                "per_layer_from_layer0": {
                    k: round(v, 4)
                    for k, v in self.predictability.per_layer_accuracy.items()
                },
            },
            "per_layer": layer_summaries,
        }

    def save_report(self, path: str | Path) -> None:
        """Save analysis report to JSON file."""
        report = self.generate_report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        report = self.generate_report()
        summary = report["summary"]
        balance = report["load_balance"]
        pred = report["predictability"]

        print("\n" + "=" * 70)
        print("MoE ROUTING ANALYSIS REPORT")
        print("=" * 70)

        print("\nConfiguration:")
        print(f"  Experts: {summary['num_experts']} (top-{summary['top_k']})")
        print(f"  Layers: {summary['num_layers']}")
        print(f"  Tokens profiled: {summary['total_tokens_profiled']:,}")

        print("\nLoad Balance Metrics:")
        print(f"  CV (coefficient of variation): {summary['avg_load_balance_cv']:.4f}")
        print(f"  Normalized entropy: {summary['avg_entropy']:.4f}")
        print(f"  Gini coefficient: {summary['avg_gini']:.4f}")

        print("\nExpert Utilization:")
        print(f"  Hot experts (>1.5x avg): {balance['num_hot']} - {balance['hot_experts'][:10]}{'...' if len(balance['hot_experts']) > 10 else ''}")
        print(f"  Cold experts (<0.5x avg): {balance['num_cold']} - {balance['cold_experts'][:10]}{'...' if len(balance['cold_experts']) > 10 else ''}")
        print(f"  Dead experts (never used): {balance['num_dead']} - {balance['dead_experts'][:10]}{'...' if len(balance['dead_experts']) > 10 else ''}")

        if balance['num_dead'] > 0:
            print(f"\n  OPTIMIZATION: {balance['num_dead']} experts can be pruned (never selected)")

        print("\nRouting Predictability:")
        if pred['predictable_from_layer'] >= 0:
            print(f"  Routing predictable from layer: {pred['predictable_from_layer']}")
            print(f"  Average prediction accuracy: {pred['avg_prediction_accuracy']:.2%}")
            print("  OPTIMIZATION: Pre-load experts based on early layer routing")
        else:
            print("  Routing not strongly predictable from early layers")

        print("\nTop Co-occurring Expert Pairs:")
        for pair in report["cooccurrence"]["top_pairs"][:5]:
            e1, e2 = pair["experts"]
            print(f"  Experts {e1} + {e2}: {pair['count']:,} co-occurrences")

        print()

    def plot_routing_heatmap(
        self,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """Plot routing pattern heatmaps.

        Creates visualizations of:
        1. Expert load distribution per layer
        2. Co-occurrence matrix
        3. Layer correlation matrix

        Args:
            output_path: Path to save the figure (PNG, PDF, etc.)
            show: If True, display the plot interactively
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Expert load per layer heatmap
        ax1 = axes[0, 0]
        load_matrix = np.zeros((self.num_layers, self.num_experts))
        for layer_idx, profile in self.layer_profiles.items():
            for expert_id, stats in profile.expert_stats.items():
                load_matrix[layer_idx, expert_id] = stats.activation_rate

        im1 = ax1.imshow(load_matrix, aspect="auto", cmap="YlOrRd")
        ax1.set_xlabel("Expert ID")
        ax1.set_ylabel("Layer")
        ax1.set_title("Expert Activation Rate per Layer")
        plt.colorbar(im1, ax=ax1, label="Activation Rate")

        # 2. Co-occurrence matrix
        ax2 = axes[0, 1]
        # Normalize for visualization
        cooc = self.cooccurrence.cooccurrence_matrix.astype(float)
        cooc_norm = cooc / cooc.max() if cooc.max() > 0 else cooc
        im2 = ax2.imshow(cooc_norm, aspect="auto", cmap="Blues")
        ax2.set_xlabel("Expert ID")
        ax2.set_ylabel("Expert ID")
        ax2.set_title("Expert Co-occurrence Matrix (normalized)")
        plt.colorbar(im2, ax=ax2, label="Co-occurrence")

        # 3. Layer correlation matrix
        ax3 = axes[1, 0]
        im3 = ax3.imshow(
            self.predictability.layer_correlations,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
        )
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Layer")
        ax3.set_title("Layer-to-Layer Routing Correlation")
        plt.colorbar(im3, ax=ax3, label="Correlation")

        # 4. Load balance metrics per layer
        ax4 = axes[1, 1]
        layers = []
        cvs = []
        entropies = []
        ginis = []

        for layer_idx, profile in self.layer_profiles.items():
            if profile.total_tokens > 0:
                layers.append(layer_idx)
                cvs.append(profile.load_balance_cv)
                entropies.append(profile.entropy)
                ginis.append(profile.gini_coefficient)

        if layers:
            x = np.arange(len(layers))
            width = 0.25
            ax4.bar(x - width, cvs, width, label="CV (lower=better)", alpha=0.8)
            ax4.bar(x, entropies, width, label="Entropy (higher=better)", alpha=0.8)
            ax4.bar(x + width, ginis, width, label="Gini (lower=better)", alpha=0.8)
            ax4.set_xlabel("Layer")
            ax4.set_ylabel("Metric Value")
            ax4.set_title("Load Balance Metrics per Layer")
            ax4.legend()
            ax4.set_xticks(x[::max(1, len(x) // 10)])  # Show every 10th label
            ax4.set_xticklabels([str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 10))])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved routing analysis plot to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def analyze_routing_from_file(
    routing_log_path: str | Path,
    num_experts: int,
    num_layers: int,
    top_k: int,
    output_dir: str | Path | None = None,
) -> MoERoutingProfiler:
    """Load routing decisions from a log file and analyze.

    Expected log format (JSON lines):
    {"layer": 0, "expert_ids": [[1, 5], [3, 7], ...], "probs": [[0.6, 0.4], ...]}

    Args:
        routing_log_path: Path to routing log file
        num_experts: Number of experts per layer
        num_layers: Number of MoE layers
        top_k: Experts per token
        output_dir: Optional directory for saving outputs

    Returns:
        Configured MoERoutingProfiler with loaded data
    """
    profiler = MoERoutingProfiler(
        num_experts=num_experts,
        num_layers=num_layers,
        top_k=top_k,
    )

    routing_log_path = Path(routing_log_path)
    with open(routing_log_path) as f:
        for line in f:
            record = json.loads(line.strip())
            layer_idx = record["layer"]
            expert_ids = np.array(record["expert_ids"])
            expert_probs = np.array(record.get("probs")) if "probs" in record else None
            profiler.record_routing(layer_idx, expert_ids, expert_probs)

    # Generate outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        profiler.save_report(output_dir / "routing_report.json")
        profiler.plot_routing_heatmap(output_dir / "routing_heatmap.png")
        profiler.print_summary()

    return profiler


def simulate_routing_for_model(
    model_name: str,
    num_samples: int = 1000,
    seed: int = 42,
) -> MoERoutingProfiler:
    """Simulate routing patterns for a known MoE model architecture.

    Useful for testing and understanding expected patterns.

    Args:
        model_name: One of "qwen3_30b", "mixtral", "glm47"
        num_samples: Number of tokens to simulate
        seed: Random seed for reproducibility

    Returns:
        MoERoutingProfiler with simulated routing data
    """
    configs = {
        "qwen3_30b": {"num_experts": 128, "num_layers": 48, "top_k": 8},
        "mixtral": {"num_experts": 8, "num_layers": 32, "top_k": 2},
        "glm47": {"num_experts": 64, "num_layers": 40, "top_k": 2},
        "deepseek_moe": {"num_experts": 64, "num_layers": 28, "top_k": 6},
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(configs.keys())}")

    config = configs[model_name]
    rng = np.random.default_rng(seed)

    profiler = MoERoutingProfiler(**config)

    # Simulate routing with some realistic patterns:
    # - Some experts are naturally "hotter" (routing weights drawn from Dirichlet)
    # - Adjacent layers have correlated routing
    # - A few experts are "dead" (very low base probability)

    for layer_idx in range(config["num_layers"]):
        # Create layer-specific expert popularity (Dirichlet gives natural skew)
        alpha = np.ones(config["num_experts"]) * 0.5
        # Make some experts more popular
        popular_experts = rng.choice(config["num_experts"], size=config["num_experts"] // 4, replace=False)
        alpha[popular_experts] *= 3.0
        # Make some experts very unpopular (candidates for pruning)
        if layer_idx % 5 == 0:  # Every 5th layer has some dead experts
            dead_experts = rng.choice(config["num_experts"], size=config["num_experts"] // 20, replace=False)
            alpha[dead_experts] = 0.01

        expert_probs_base = rng.dirichlet(alpha)

        # Generate routing for this layer
        expert_ids = []
        expert_probs = []

        for _ in range(num_samples):
            # Add noise to base probabilities
            token_probs = expert_probs_base + rng.normal(0, 0.01, config["num_experts"])
            token_probs = np.clip(token_probs, 0, None)
            token_probs /= token_probs.sum()

            # Select top-k experts
            top_k_idx = np.argsort(token_probs)[-config["top_k"] :][::-1]
            top_k_probs = token_probs[top_k_idx]
            top_k_probs /= top_k_probs.sum()  # Renormalize

            expert_ids.append(top_k_idx)
            expert_probs.append(top_k_probs)

        profiler.record_routing(
            layer_idx,
            np.array(expert_ids),
            np.array(expert_probs),
        )

    return profiler


def main() -> int:
    """CLI entry point for MoE routing analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MoE routing patterns for optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Analyze from routing log file
    python -m metal_marlin.analysis.moe_routing --log routing.jsonl \\
        --num-experts 64 --num-layers 40 --top-k 2 --output results/

    # Simulate and analyze Qwen3-30B routing
    python -m metal_marlin.analysis.moe_routing --simulate qwen3_30b \\
        --samples 5000 --output results/

    # Simulate Mixtral routing
    python -m metal_marlin.analysis.moe_routing --simulate mixtral
""",
    )

    parser.add_argument(
        "--log",
        type=Path,
        help="Path to routing log file (JSON lines)",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        choices=["qwen3_30b", "mixtral", "glm47", "deepseek_moe"],
        help="Simulate routing for a known model architecture",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of tokens to simulate (default: 1000)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        help="Number of experts per layer (required for --log)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of MoE layers (required for --log)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Experts per token (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for report and plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation (default: 42)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    if args.log:
        if not args.num_experts or not args.num_layers:
            parser.error("--num-experts and --num-layers required when using --log")

        profiler = analyze_routing_from_file(
            routing_log_path=args.log,
            num_experts=args.num_experts,
            num_layers=args.num_layers,
            top_k=args.top_k,
            output_dir=args.output,
        )

    elif args.simulate:
        print(f"Simulating routing patterns for {args.simulate}...")
        profiler = simulate_routing_for_model(
            model_name=args.simulate,
            num_samples=args.samples,
            seed=args.seed,
        )

    else:
        parser.error("Either --log or --simulate is required")
        return 1

    # Print summary
    profiler.print_summary()

    # Save outputs
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        profiler.save_report(args.output / "routing_report.json")
        print(f"Report saved to: {args.output / 'routing_report.json'}")

        if not args.no_plot:
            profiler.plot_routing_heatmap(args.output / "routing_heatmap.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
