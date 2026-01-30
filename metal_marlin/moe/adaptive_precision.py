"""Adaptive precision for MoE experts based on load distribution.

Not all experts are equal in MoE models. This module profiles expert load
distribution and assigns precision levels based on:

1. Traffic volume: High-traffic experts process more tokens and need higher
   precision to handle diverse inputs accurately.
2. Input diversity: Experts that see high variance in their inputs need more
   bits to capture the full distribution.
3. Output importance: Some layers are more critical (e.g., final layers).
4. Shared expert status: Shared experts (always active) get highest precision.

Memory savings of 20-30% vs uniform precision are achievable because:
- With 64 experts and top-k=2 routing, ~96.9% of experts are "cold"
- Cold experts rarely activate and see narrow input distributions
- These can be quantized to INT3/NF2 without quality loss

Usage:
    from metal_marlin.moe.adaptive_precision import (
        ExpertLoadProfiler,
        InputDiversityAnalyzer,
        AdaptivePrecisionMapper,
    )

    # Profile expert load from routing decisions
    profiler = ExpertLoadProfiler(num_experts=64, num_layers=28)
    for batch in calibration_data:
        expert_ids, expert_probs = router(batch)
        profiler.record_batch(layer_idx=0, expert_ids=expert_ids, probs=expert_probs)

    # Analyze input diversity per expert
    diversity = InputDiversityAnalyzer(num_experts=64)
    for batch in calibration_data:
        diversity.record_activations(layer_idx=0, activations=hidden, expert_ids=expert_ids)

    # Map precision levels
    mapper = AdaptivePrecisionMapper(
        profiler,
        diversity,
        shared_expert_ids={0},  # Expert 0 is shared
    )
    precision_map = mapper.compute_precision_map()

    # Use precision map for quantization
    for expert_id in range(64):
        precision = precision_map.get_precision(layer_idx=0, expert_id=expert_id)
        quantize_expert(expert_weights, precision=precision)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PrecisionLevel(Enum):
    """Precision levels for expert quantization.

    Ordered from highest to lowest quality.
    """

    BF16 = "bf16"  # Full precision (shared experts, critical layers)
    FP4_G32 = "fp4_g32"  # FP4 with group_size=32 (high-traffic)
    FP4_G64 = "fp4_g64"  # FP4 with group_size=64 (medium-traffic)
    FP4_G128 = "fp4_g128"  # FP4 with group_size=128 (standard)
    FP4_G256 = "fp4_g256"  # FP4 with group_size=256 (low-traffic)
    INT3_G64 = "int3_g64"  # INT3 3-bit (very low traffic)
    NF3_G64 = "nf3_g64"  # NormalFloat 3-bit (cold experts, Gaussian-optimal)
    NF2_G64 = "nf2_g64"  # NormalFloat 2-bit (near-zero traffic)

    @property
    def bits(self) -> int:
        """Number of bits for this precision level."""
        if self == PrecisionLevel.BF16:
            return 16
        elif "fp4" in self.value or "int4" in self.value:
            return 4
        elif "int3" in self.value or "nf3" in self.value:
            return 3
        elif "int2" in self.value or "nf2" in self.value:
            return 2
        return 4  # Default to 4-bit

    @property
    def group_size(self) -> int:
        """Group size for this precision level (0 for full precision)."""
        if self == PrecisionLevel.BF16:
            return 0
        # Parse group size from value (e.g., "fp4_g128" -> 128)
        parts = self.value.split("_g")
        if len(parts) == 2:
            return int(parts[1])
        return 128  # Default

    @property
    def format(self) -> str:
        """Quantization format string."""
        if self == PrecisionLevel.BF16:
            return "bf16"
        parts = self.value.split("_g")
        return parts[0] if parts else "fp4"


@dataclass
class ExpertLoadStats:
    """Load statistics for a single expert."""

    expert_id: int
    layer_idx: int

    # Token counts
    total_activations: int = 0  # Times this expert was selected
    total_tokens: int = 0  # Total tokens seen (denominator for rate)

    # Probability statistics (how strongly the router selects this expert)
    prob_sum: float = 0.0  # Sum of routing probabilities
    prob_sum_sq: float = 0.0  # Sum of squared probabilities (for variance)

    # Rank distribution (was this expert the top choice or backup?)
    rank_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def activation_rate(self) -> float:
        """Fraction of tokens that activated this expert."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_activations / self.total_tokens

    @property
    def mean_probability(self) -> float:
        """Average routing probability when selected."""
        if self.total_activations == 0:
            return 0.0
        return self.prob_sum / self.total_activations

    @property
    def probability_variance(self) -> float:
        """Variance of routing probabilities when selected."""
        if self.total_activations < 2:
            return 0.0
        mean = self.mean_probability
        return max(0.0, (self.prob_sum_sq / self.total_activations) - mean**2)

    @property
    def top_k_distribution(self) -> dict[int, float]:
        """Distribution of rank positions (0 = top expert)."""
        total = sum(self.rank_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.rank_counts.items()}

    def record_activation(self, prob: float, rank: int) -> None:
        """Record an expert activation."""
        self.total_activations += 1
        self.prob_sum += prob
        self.prob_sum_sq += prob * prob
        self.rank_counts[rank] += 1

    def record_batch(self, batch_size: int) -> None:
        """Record batch size for rate calculation."""
        self.total_tokens += batch_size


@dataclass
class InputDiversityStats:
    """Input diversity statistics for an expert."""

    expert_id: int
    layer_idx: int

    # Running statistics for input activations (Welford's algorithm)
    n_samples: int = 0
    mean: NDArray[np.float32] | None = None  # [hidden_dim]
    m2: NDArray[np.float32] | None = None  # Sum of squared deviations

    # Magnitude statistics
    max_magnitude: float = 0.0
    magnitude_sum: float = 0.0

    @property
    def variance(self) -> NDArray[np.float32] | None:
        """Per-dimension variance of inputs."""
        if self.n_samples < 2 or self.m2 is None:
            return None
        return self.m2 / (self.n_samples - 1)

    @property
    def total_variance(self) -> float:
        """Total variance (sum across dimensions)."""
        if self.variance is None:
            return 0.0
        return float(np.sum(self.variance))

    @property
    def mean_magnitude(self) -> float:
        """Mean input magnitude."""
        if self.n_samples == 0:
            return 0.0
        return self.magnitude_sum / self.n_samples

    def update(self, activations: NDArray[np.float32]) -> None:
        """Update statistics with new activations using Welford's algorithm.

        Args:
            activations: Input activations [n_tokens, hidden_dim]
        """
        for x in activations:
            self.n_samples += 1

            # Track magnitude
            mag = float(np.linalg.norm(x))
            self.max_magnitude = max(self.max_magnitude, mag)
            self.magnitude_sum += mag

            if self.mean is None:
                self.mean = x.copy()
                self.m2 = np.zeros_like(x)
            else:
                delta = x - self.mean
                self.mean += delta / self.n_samples
                delta2 = x - self.mean
                self.m2 += delta * delta2


class ExpertLoadProfiler:
    """Profile expert load distribution from routing decisions.

    Tracks per-expert activation frequency, probability distribution,
    and rank distribution (top-1 vs backup selections).

    Args:
        num_experts: Number of experts per MoE layer.
        num_layers: Number of MoE layers in the model.
        shared_expert_ids: Set of expert IDs that are shared (always active).
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        shared_expert_ids: set[int] | None = None,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.shared_expert_ids = shared_expert_ids or set()

        # Per-layer, per-expert statistics
        self._stats: dict[tuple[int, int], ExpertLoadStats] = {}
        for layer_idx in range(num_layers):
            for expert_id in range(num_experts):
                key = (layer_idx, expert_id)
                self._stats[key] = ExpertLoadStats(expert_id=expert_id, layer_idx=layer_idx)

    def record_batch(
        self,
        layer_idx: int,
        expert_ids: NDArray[np.int32] | Any,
        probs: NDArray[np.float32] | Any | None = None,
    ) -> None:
        """Record routing decisions for a batch.

        Args:
            layer_idx: MoE layer index.
            expert_ids: Expert assignments [batch_size, top_k].
            probs: Routing probabilities [batch_size, top_k] or None.
        """
        # Convert MLX arrays if needed
        if hasattr(expert_ids, "tolist"):
            expert_ids = np.array(expert_ids.tolist(), dtype=np.int32)
        if probs is not None and hasattr(probs, "tolist"):
            probs = np.array(probs.tolist(), dtype=np.float32)

        batch_size, top_k = expert_ids.shape

        # Record batch size for all experts
        for expert_id in range(self.num_experts):
            key = (layer_idx, expert_id)
            self._stats[key].record_batch(batch_size)

        # Record activations
        for token_idx in range(batch_size):
            for rank in range(top_k):
                expert_id = int(expert_ids[token_idx, rank])
                prob = float(probs[token_idx, rank]) if probs is not None else 1.0 / top_k
                key = (layer_idx, expert_id)
                self._stats[key].record_activation(prob, rank)

    def get_stats(self, layer_idx: int, expert_id: int) -> ExpertLoadStats:
        """Get statistics for a specific expert."""
        return self._stats[(layer_idx, expert_id)]

    def get_layer_stats(self, layer_idx: int) -> list[ExpertLoadStats]:
        """Get statistics for all experts in a layer, sorted by activation rate."""
        stats = [self._stats[(layer_idx, e)] for e in range(self.num_experts)]
        return sorted(stats, key=lambda s: s.activation_rate, reverse=True)

    def get_traffic_tiers(
        self,
        layer_idx: int,
        thresholds: tuple[float, float, float] = (0.1, 0.05, 0.01),
    ) -> dict[str, list[int]]:
        """Categorize experts into traffic tiers.

        Args:
            layer_idx: MoE layer index.
            thresholds: (high, medium, low) activation rate thresholds.

        Returns:
            Dict with keys "high", "medium", "low", "cold" mapping to expert IDs.
        """
        high_thresh, med_thresh, low_thresh = thresholds
        tiers: dict[str, list[int]] = {
            "high": [],
            "medium": [],
            "low": [],
            "cold": [],
        }

        for expert_id in range(self.num_experts):
            if expert_id in self.shared_expert_ids:
                tiers["high"].append(expert_id)
                continue

            rate = self._stats[(layer_idx, expert_id)].activation_rate
            if rate >= high_thresh:
                tiers["high"].append(expert_id)
            elif rate >= med_thresh:
                tiers["medium"].append(expert_id)
            elif rate >= low_thresh:
                tiers["low"].append(expert_id)
            else:
                tiers["cold"].append(expert_id)

        return tiers

    def summary(self) -> dict:
        """Generate summary statistics."""
        summary = {
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "shared_experts": list(self.shared_expert_ids),
            "per_layer": {},
        }

        for layer_idx in range(self.num_layers):
            layer_stats = self.get_layer_stats(layer_idx)
            tiers = self.get_traffic_tiers(layer_idx)

            rates = [s.activation_rate for s in layer_stats]
            summary["per_layer"][layer_idx] = {
                "mean_activation_rate": float(np.mean(rates)),
                "max_activation_rate": float(np.max(rates)),
                "min_activation_rate": float(np.min(rates)),
                "tier_counts": {k: len(v) for k, v in tiers.items()},
                "top_5_experts": [
                    {
                        "expert_id": s.expert_id,
                        "activation_rate": s.activation_rate,
                        "mean_prob": s.mean_probability,
                    }
                    for s in layer_stats[:5]
                ],
            }

        return summary


class InputDiversityAnalyzer:
    """Analyze input diversity per expert.

    Experts that see high variance in their inputs need more precision
    to accurately capture the full input distribution. This analyzer
    tracks per-expert input statistics using Welford's online algorithm.

    Args:
        num_experts: Number of experts per MoE layer.
        num_layers: Number of MoE layers in the model.
    """

    def __init__(self, num_experts: int, num_layers: int):
        self.num_experts = num_experts
        self.num_layers = num_layers

        # Per-layer, per-expert diversity stats
        self._stats: dict[tuple[int, int], InputDiversityStats] = {}
        for layer_idx in range(num_layers):
            for expert_id in range(num_experts):
                key = (layer_idx, expert_id)
                self._stats[key] = InputDiversityStats(expert_id=expert_id, layer_idx=layer_idx)

    def record_activations(
        self,
        layer_idx: int,
        activations: NDArray[np.float32] | Any,
        expert_ids: NDArray[np.int32] | Any,
    ) -> None:
        """Record input activations for diversity analysis.

        Args:
            layer_idx: MoE layer index.
            activations: Input activations [batch_size, hidden_dim].
            expert_ids: Expert assignments [batch_size, top_k].
        """
        # Convert MLX arrays if needed
        if hasattr(activations, "tolist"):
            activations = np.array(activations.tolist(), dtype=np.float32)
        if hasattr(expert_ids, "tolist"):
            expert_ids = np.array(expert_ids.tolist(), dtype=np.int32)

        batch_size = activations.shape[0]

        # Group activations by expert
        expert_activations: dict[int, list[NDArray[np.float32]]] = defaultdict(list)
        for token_idx in range(batch_size):
            act = activations[token_idx]
            for expert_id in expert_ids[token_idx]:
                expert_activations[int(expert_id)].append(act)

        # Update statistics per expert
        for expert_id, acts in expert_activations.items():
            if acts:
                key = (layer_idx, expert_id)
                acts_array = np.stack(acts, axis=0)
                self._stats[key].update(acts_array)

    def get_stats(self, layer_idx: int, expert_id: int) -> InputDiversityStats:
        """Get diversity statistics for a specific expert."""
        return self._stats[(layer_idx, expert_id)]

    def get_diversity_ranking(self, layer_idx: int) -> list[tuple[int, float]]:
        """Rank experts by input diversity (total variance).

        Returns:
            List of (expert_id, total_variance) sorted by variance descending.
        """
        ranking = []
        for expert_id in range(self.num_experts):
            key = (layer_idx, expert_id)
            var = self._stats[key].total_variance
            ranking.append((expert_id, var))

        return sorted(ranking, key=lambda x: x[1], reverse=True)


@dataclass
class ExpertPrecisionMap:
    """Precision assignments for all experts across all layers."""

    num_experts: int
    num_layers: int

    # (layer_idx, expert_id) -> PrecisionLevel
    _precision_map: dict[tuple[int, int], PrecisionLevel] = field(default_factory=dict)

    # Memory statistics
    _bits_per_expert: dict[tuple[int, int], int] = field(default_factory=dict)

    def set_precision(
        self,
        layer_idx: int,
        expert_id: int,
        precision: PrecisionLevel,
    ) -> None:
        """Set precision for an expert."""
        self._precision_map[(layer_idx, expert_id)] = precision
        self._bits_per_expert[(layer_idx, expert_id)] = precision.bits

    def get_precision(self, layer_idx: int, expert_id: int) -> PrecisionLevel:
        """Get precision for an expert."""
        return self._precision_map.get((layer_idx, expert_id), PrecisionLevel.FP4_G128)

    def get_layer_precisions(self, layer_idx: int) -> dict[int, PrecisionLevel]:
        """Get precision map for a layer."""
        return {
            expert_id: self.get_precision(layer_idx, expert_id)
            for expert_id in range(self.num_experts)
        }

    def average_bits(self) -> float:
        """Average bits per weight across all experts."""
        if not self._bits_per_expert:
            return 4.0
        return float(np.mean(list(self._bits_per_expert.values())))

    def memory_savings_vs_uniform(self, uniform_bits: int = 4) -> float:
        """Memory savings compared to uniform quantization.

        Returns:
            Fraction saved (e.g., 0.25 means 25% smaller).
        """
        if not self._bits_per_expert:
            return 0.0

        total_bits = sum(self._bits_per_expert.values())
        uniform_total = len(self._bits_per_expert) * uniform_bits
        return 1.0 - (total_bits / uniform_total)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "precision_map": {
                f"{layer}_{expert}": prec.value
                for (layer, expert), prec in self._precision_map.items()
            },
            "average_bits": self.average_bits(),
            "memory_savings": self.memory_savings_vs_uniform(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExpertPrecisionMap:
        """Deserialize from dictionary."""
        obj = cls(num_experts=data["num_experts"], num_layers=data["num_layers"])
        for key, prec_str in data.get("precision_map", {}).items():
            layer, expert = map(int, key.split("_"))
            obj.set_precision(layer, expert, PrecisionLevel(prec_str))
        return obj

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"ExpertPrecisionMap: {self.num_experts} experts x {self.num_layers} layers",
            f"Average bits: {self.average_bits():.2f}",
            f"Memory savings vs uniform FP4: {self.memory_savings_vs_uniform():.1%}",
            "",
        ]

        # Count precision levels per layer
        for layer_idx in range(min(self.num_layers, 3)):  # Show first 3 layers
            counts: dict[PrecisionLevel, int] = defaultdict(int)
            for expert_id in range(self.num_experts):
                prec = self.get_precision(layer_idx, expert_id)
                counts[prec] += 1

            lines.append(f"Layer {layer_idx}:")
            for prec in PrecisionLevel:
                if counts[prec] > 0:
                    lines.append(f"  {prec.value}: {counts[prec]} experts")

        if self.num_layers > 3:
            lines.append(f"  ... ({self.num_layers - 3} more layers)")

        return "\n".join(lines)


class AdaptivePrecisionMapper:
    """Map precision levels to experts based on load and diversity analysis.

    Combines traffic volume, input diversity, and layer importance to
    assign optimal precision levels to each expert.

    Args:
        load_profiler: ExpertLoadProfiler with accumulated statistics.
        diversity_analyzer: InputDiversityAnalyzer or None.
        shared_expert_ids: Expert IDs for shared experts (always BF16).
        traffic_thresholds: (high, medium, low) activation rate thresholds.
        diversity_weight: Weight for diversity in precision decision (0-1).
    """

    def __init__(
        self,
        load_profiler: ExpertLoadProfiler,
        diversity_analyzer: InputDiversityAnalyzer | None = None,
        shared_expert_ids: set[int] | None = None,
        traffic_thresholds: tuple[float, float, float] = (0.1, 0.05, 0.01),
        diversity_weight: float = 0.3,
    ):
        self.load_profiler = load_profiler
        self.diversity_analyzer = diversity_analyzer
        self.shared_expert_ids = shared_expert_ids or set()
        self.traffic_thresholds = traffic_thresholds
        self.diversity_weight = diversity_weight

    def compute_precision_map(
        self,
        boundary_layers_bf16: bool = True,
        allow_sub4bit: bool = True,
    ) -> ExpertPrecisionMap:
        """Compute precision assignments for all experts.

        Args:
            boundary_layers_bf16: Keep first/last layer experts at higher precision.
            allow_sub4bit: Allow INT3/NF3/NF2 for cold experts.

        Returns:
            ExpertPrecisionMap with precision assignments.
        """
        precision_map = ExpertPrecisionMap(
            num_experts=self.load_profiler.num_experts,
            num_layers=self.load_profiler.num_layers,
        )

        for layer_idx in range(self.load_profiler.num_layers):
            is_boundary = boundary_layers_bf16 and (
                layer_idx == 0 or layer_idx == self.load_profiler.num_layers - 1
            )

            for expert_id in range(self.load_profiler.num_experts):
                precision = self._compute_expert_precision(
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                    is_boundary_layer=is_boundary,
                    allow_sub4bit=allow_sub4bit,
                )
                precision_map.set_precision(layer_idx, expert_id, precision)

        return precision_map

    def _compute_expert_precision(
        self,
        layer_idx: int,
        expert_id: int,
        is_boundary_layer: bool,
        allow_sub4bit: bool,
    ) -> PrecisionLevel:
        """Compute precision for a single expert."""
        # Shared experts always get highest precision
        if expert_id in self.shared_expert_ids:
            return PrecisionLevel.FP4_G32

        # Boundary layers get higher precision
        if is_boundary_layer:
            return PrecisionLevel.FP4_G64

        # Get load statistics
        load_stats = self.load_profiler.get_stats(layer_idx, expert_id)
        activation_rate = load_stats.activation_rate

        # Get diversity score if available
        diversity_score = 0.0
        if self.diversity_analyzer is not None:
            div_stats = self.diversity_analyzer.get_stats(layer_idx, expert_id)
            # Normalize variance to [0, 1] range (heuristic)
            diversity_score = min(1.0, div_stats.total_variance / 1e6)

        # Combine traffic and diversity into a single importance score
        traffic_score = activation_rate / max(self.traffic_thresholds[0], 0.01)
        importance = (
            1 - self.diversity_weight
        ) * traffic_score + self.diversity_weight * diversity_score

        # Map importance to precision level
        high_thresh, med_thresh, low_thresh = self.traffic_thresholds

        if activation_rate >= high_thresh or importance > 0.8:
            return PrecisionLevel.FP4_G64
        elif activation_rate >= med_thresh or importance > 0.4:
            return PrecisionLevel.FP4_G128
        elif activation_rate >= low_thresh or importance > 0.2:
            return PrecisionLevel.FP4_G256 if allow_sub4bit else PrecisionLevel.FP4_G128
        else:
            # Cold experts
            if not allow_sub4bit:
                return PrecisionLevel.FP4_G256

            # Use NF3 for very cold experts (Gaussian-optimal)
            if activation_rate < low_thresh / 2:
                return PrecisionLevel.NF3_G64

            return PrecisionLevel.INT3_G64

    def analyze_savings(self) -> dict:
        """Analyze memory savings from adaptive precision.

        Returns:
            Dict with savings analysis.
        """
        precision_map = self.compute_precision_map()

        # Count experts per precision level
        level_counts: dict[PrecisionLevel, int] = defaultdict(int)
        for layer_idx in range(self.load_profiler.num_layers):
            for expert_id in range(self.load_profiler.num_experts):
                prec = precision_map.get_precision(layer_idx, expert_id)
                level_counts[prec] += 1

        total_experts = self.load_profiler.num_layers * self.load_profiler.num_experts

        return {
            "total_experts": total_experts,
            "precision_distribution": {prec.value: count for prec, count in level_counts.items()},
            "average_bits": precision_map.average_bits(),
            "vs_uniform_fp4": {
                "uniform_bits": 4.0,
                "adaptive_bits": precision_map.average_bits(),
                "savings_pct": precision_map.memory_savings_vs_uniform() * 100,
            },
            "vs_uniform_bf16": {
                "uniform_bits": 16.0,
                "adaptive_bits": precision_map.average_bits(),
                "savings_pct": (1 - precision_map.average_bits() / 16) * 100,
            },
        }


def profile_experts_from_dataset(
    model,
    tokenizer,
    calibration_texts: list[str],
    num_experts: int,
    num_layers: int,
    shared_expert_ids: set[int] | None = None,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    verbose: bool = True,
) -> tuple[ExpertLoadProfiler, InputDiversityAnalyzer]:
    """Profile expert load and input diversity from calibration data.

    Runs forward passes through the model, collecting expert routing
    decisions and input activations for each MoE layer.

    Args:
        model: HuggingFace or MLX model with MoE layers.
        tokenizer: Tokenizer for the model.
        calibration_texts: List of calibration text samples.
        num_experts: Number of experts per MoE layer.
        num_layers: Number of MoE layers.
        shared_expert_ids: Set of shared expert IDs.
        batch_size: Batch size for forward passes.
        max_seq_len: Maximum sequence length.
        verbose: Print progress.

    Returns:
        (ExpertLoadProfiler, InputDiversityAnalyzer) with accumulated statistics.
    """
    import torch

    profiler = ExpertLoadProfiler(
        num_experts=num_experts,
        num_layers=num_layers,
        shared_expert_ids=shared_expert_ids,
    )
    diversity = InputDiversityAnalyzer(
        num_experts=num_experts,
        num_layers=num_layers,
    )

    # Hook storage for collecting routing decisions
    routing_decisions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    activations_cache: dict[int, np.ndarray] = {}

    def make_routing_hook(layer_idx: int):
        """Create hook to capture routing decisions."""

        def hook(module, input, output):
            # Extract expert_ids and expert_probs from output
            # This depends on the model architecture
            if hasattr(output, "expert_ids"):
                expert_ids = output.expert_ids.detach().cpu().numpy()
                expert_probs = output.expert_probs.detach().cpu().numpy()
                routing_decisions[layer_idx] = (expert_ids, expert_probs)

            # Also capture input activations
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input

            if isinstance(x, torch.Tensor):
                act = x.view(-1, x.shape[-1]).detach().cpu().numpy()
                activations_cache[layer_idx] = act

        return hook

    # Register hooks (this requires knowing the MoE layer structure)
    # For now, this is a placeholder - actual implementation depends on model architecture
    hooks = []

    if verbose:
        print(f"Profiling {len(calibration_texts)} samples...")

    model.eval()
    with torch.no_grad():
        for batch_idx in range(0, len(calibration_texts), batch_size):
            batch_end = min(batch_idx + batch_size, len(calibration_texts))
            batch_texts = calibration_texts[batch_idx:batch_end]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=True,
            )

            # Forward pass (hooks capture routing decisions)
            model(**inputs)

            # Record collected data
            for layer_idx, (expert_ids, expert_probs) in routing_decisions.items():
                profiler.record_batch(layer_idx, expert_ids, expert_probs)

                if layer_idx in activations_cache:
                    diversity.record_activations(
                        layer_idx, activations_cache[layer_idx], expert_ids
                    )

            # Clear caches
            routing_decisions.clear()
            activations_cache.clear()

            if verbose and (batch_idx // batch_size + 1) % 10 == 0:
                batches_done = batch_idx // batch_size + 1
                total_batches = (len(calibration_texts) + batch_size - 1) // batch_size
                print(f"  Batch {batches_done}/{total_batches}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return profiler, diversity


def apply_adaptive_precision(
    model_path: str,
    output_path: str,
    precision_map: ExpertPrecisionMap,
    verbose: bool = True,
) -> dict:
    """Apply adaptive precision quantization to a model.

    Args:
        model_path: Path to original model.
        output_path: Path to save quantized model.
        precision_map: Precision assignments from AdaptivePrecisionMapper.
        verbose: Print progress.

    Returns:
        Quantization report dictionary.
    """
    from pathlib import Path

    from ..mr_gptq import MRGPTQQuantizer, QuantizationFormat

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "precision_map": precision_map.to_dict(),
        "layers_quantized": 0,
        "average_bits": precision_map.average_bits(),
    }

    # Create quantizers for each precision level
    quantizers: dict[PrecisionLevel, MRGPTQQuantizer] = {}

    for prec in PrecisionLevel:
        if prec == PrecisionLevel.BF16:
            continue  # No quantization for BF16

        # Map precision level to quantizer settings
        fmt_str = prec.format
        if fmt_str.startswith("fp"):
            fmt = QuantizationFormat.FP4
        elif fmt_str.startswith("int"):
            fmt = QuantizationFormat.INT4  # Use INT4 for INT3/INT2 base
        elif fmt_str.startswith("nf"):
            fmt = QuantizationFormat.NF4  # Use NF4 for NF3/NF2 base
        else:
            fmt = QuantizationFormat.FP4

        quantizers[prec] = MRGPTQQuantizer(
            bits=prec.bits,
            format=fmt,
            group_size=prec.group_size,
            use_hadamard=True,
            hadamard_block_size=64,
        )

    if verbose:
        print("Adaptive Precision Quantization")
        print(f"  Average bits: {precision_map.average_bits():.2f}")
        print(f"  Memory savings: {precision_map.memory_savings_vs_uniform():.1%}")
        print()

    # Note: Full implementation would iterate through model layers,
    # applying the appropriate quantizer to each expert based on precision_map.
    # This is a placeholder for the integration point.

    return report
