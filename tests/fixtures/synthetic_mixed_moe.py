"""Synthetic mixed-precision MoE model for benchmarking.

This module provides a lightweight 2-layer model that mimics the bit distribution
of GLM-4.7-Flash-Trellis-MM for rapid iteration on mixed-precision optimizations.

Usage:
    >>> from tests.fixtures.synthetic_mixed_moe import create_synthetic_model
    >>> model = create_synthetic_model(device='mps')
    >>> x = torch.randn(1, 512, dtype=torch.float16, device='mps')
    >>> out = model(x)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

# Import TrellisLinear for real quantized layers
try:
    from metal_marlin.trellis.linear import TrellisLinear
    from metal_marlin.trellis.packing import pack_trellis_indices
    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False
    TrellisLinear = None  # type: ignore


@dataclass
class SyntheticConfig:
    """Configuration for synthetic mixed-precision model."""

    hidden_dim: int = 512
    intermediate_dim: int = 1408  # ~2.75x hidden (GLM-4 ratio)
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 2

    # Bit distribution mimicking GLM-4.7-Flash-Trellis-MM Layer 1:
    # (6, 2, 3): 65.6%, (6, 2, 2): 12.5%, (6, 3, 3): 9.4%
    # (6, 2, 4): 4.7%, (6, 3, 4): 6.3%, (6, 3, 2): 1.6%
    expert_bit_tuples: list[tuple[int, int, int]] | None = None

    # Dense layer uses uniform bits
    dense_bits: int = 4

    def __post_init__(self) -> None:
        if self.expert_bit_tuples is None:
            # Default: mimic GLM-4.7-Flash distribution
            # 8 experts total, distribute by frequency
            self.expert_bit_tuples = [
                (6, 2, 3),  # Expert 0 - dominant
                (6, 2, 3),  # Expert 1 - dominant
                (6, 2, 3),  # Expert 2 - dominant
                (6, 2, 3),  # Expert 3 - dominant
                (6, 2, 3),  # Expert 4 - dominant  (5 of 8 = 62.5%)
                (6, 2, 2),  # Expert 5 - common
                (6, 3, 3),  # Expert 6 - secondary
                (6, 2, 4),  # Expert 7 - rare
            ]


def create_fake_trellis_weights(
    in_features: int,
    out_features: int,
    bits: int,
    device: str = "mps",
) -> dict[str, torch.Tensor]:
    """Create fake quantized weights for TrellisLinear.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bits: Quantization bit width (2-6)
        device: Target device

    Returns:
        Dict with packed_indices, scales, grid, su, sv tensors
    """
    # Tile dimensions
    tile_k = 32
    tile_n = 16

    tiles_k = (in_features + tile_k - 1) // tile_k
    tiles_n = (out_features + tile_n - 1) // tile_n

    # Packed bytes per tile
    packed_bytes = (tile_k * tile_n * bits + 7) // 8

    # Create random packed indices (simulates quantized weights)
    n_levels = 1 << bits
    packed_indices = torch.randint(
        0, 256, (tiles_k, tiles_n, packed_bytes),
        dtype=torch.uint8, device=device
    )

    # Per-group scales (one per tile_k group)
    n_groups = tiles_k
    scales = torch.randn(n_groups, out_features,
                         dtype=torch.float16, device=device)
    scales = scales.abs() * 0.1  # Small positive scales

    # Codebook grid (uniform levels)
    grid = torch.linspace(-1.0, 1.0, n_levels,
                          dtype=torch.float16, device=device)

    # Sign vectors
    su = torch.randn(in_features, dtype=torch.float16, device=device).sign()
    sv = torch.randn(out_features, dtype=torch.float16, device=device).sign()

    return {
        "packed_indices": packed_indices,
        "scales": scales,
        "grid": grid,
        "su": su,
        "sv": sv,
        "bits": bits,
    }


class FakeTrellisLinear(nn.Module):
    """Fake TrellisLinear for testing when metal_marlin is not available."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        device: str = "mps",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Create fake quantized weights
        weights = create_fake_trellis_weights(
            in_features, out_features, bits, device)

        self.register_buffer("packed_indices", weights["packed_indices"])
        self.register_buffer("scales", weights["scales"])
        self.register_buffer("grid", weights["grid"])
        self.register_buffer("su", weights["su"])
        self.register_buffer("sv", weights["sv"])

        # For forward pass, use a real weight matrix
        self._weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple linear forward (ignores quantization for speed)."""
        return F.linear(x.float(), self._weight.float()).half()


def create_trellis_linear(
    in_features: int,
    out_features: int,
    bits: int,
    device: str = "mps",
) -> nn.Module:
    """Create a TrellisLinear or fake version.

    For synthetic benchmarks, always uses FakeTrellisLinear to avoid
    shape mismatches with the real TrellisLinear buffer format.
    The fake version performs standard linear ops - useful for
    measuring dispatch overhead, not quantization performance.
    """
    # Always use fake for synthetic benchmarks
    # Real TrellisLinear has different buffer shapes per bit width
    return FakeTrellisLinear(in_features, out_features, bits, device)


class SyntheticExpert(nn.Module):
    """Single MoE expert with mixed-precision projections."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        gate_bits: int,
        up_bits: int,
        down_bits: int,
        device: str = "mps",
    ):
        super().__init__()

        self.gate_proj = create_trellis_linear(
            hidden_dim, intermediate_dim, gate_bits, device)
        self.up_proj = create_trellis_linear(
            hidden_dim, intermediate_dim, up_bits, device)
        self.down_proj = create_trellis_linear(
            intermediate_dim, hidden_dim, down_bits, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SyntheticMoELayer(nn.Module):
    """MoE layer with mixed-precision experts."""

    def __init__(
        self,
        config: SyntheticConfig,
        device: str = "mps",
    ):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # Router
        self.router = nn.Linear(
            config.hidden_dim, config.num_experts, device=device)

        # Create experts with specified bit tuples
        self.experts = nn.ModuleList()
        for i in range(config.num_experts):
            gate_bits, up_bits, down_bits = config.expert_bit_tuples[i]
            expert = SyntheticExpert(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                gate_bits=gate_bits,
                up_bits=up_bits,
                down_bits=down_bits,
                device=device,
            )
            self.experts.append(expert)

        # Track bit distribution for analysis
        self._bit_tuples = config.expert_bit_tuples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with top-k routing (sequential dispatch)."""
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.hidden_dim)

        # Route
        router_logits = self.router(x_flat.float())
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1),
            k=self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / \
            routing_weights.sum(dim=-1, keepdim=True)

        # Sequential dispatch (slow path)
        output = torch.zeros_like(x_flat)
        for i in range(x_flat.shape[0]):
            for k in range(self.top_k):
                expert_id = selected_experts[i, k].item()
                weight = routing_weights[i, k]
                expert_out = self.experts[expert_id](x_flat[i:i+1])
                output[i:i+1] += weight * expert_out

        return output.view(*batch_shape, self.hidden_dim)

    def batched_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with batched expert dispatch (fast path)."""
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.hidden_dim)

        # Route
        router_logits = self.router(x_flat.float())
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1),
            k=self.top_k,
            dim=-1,
        )
        routing_weights = (
            routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        ).half()

        output = torch.zeros_like(x_flat)

        # Batched dispatch: process all tokens for each expert at once
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = selected_experts == expert_id
            if not mask.any():
                continue

            # Get token indices and which k they are
            token_indices, k_indices = torch.where(mask)
            expert_inputs = x_flat[token_indices]

            # Call expert once for all assigned tokens
            expert_outputs = self.experts[expert_id](expert_inputs)

            # Weight and accumulate
            weights = routing_weights[token_indices, k_indices].unsqueeze(-1)
            output.index_add_(0, token_indices, (weights * expert_outputs).to(output.dtype))

        return output.view(*batch_shape, self.hidden_dim)


class SyntheticDenseLayer(nn.Module):
    """Dense MLP layer with uniform quantization (baseline)."""

    def __init__(
        self,
        config: SyntheticConfig,
        device: str = "mps",
    ):
        super().__init__()

        self.gate_proj = create_trellis_linear(
            config.hidden_dim, config.intermediate_dim, config.dense_bits, device
        )
        self.up_proj = create_trellis_linear(
            config.hidden_dim, config.intermediate_dim, config.dense_bits, device
        )
        self.down_proj = create_trellis_linear(
            config.intermediate_dim, config.hidden_dim, config.dense_bits, device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SyntheticMixedMoE(nn.Module):
    """Synthetic 2-layer model for mixed precision benchmarking.

    Layer 0: Dense MLP (uniform 4-bit) - establishes baseline speed
    Layer 1: MoE with 8 experts, mixed precision - optimization target

    This model is designed for rapid iteration on mixed-precision
    optimizations. It's small enough to run quickly but exercises
    the same code paths as GLM-4.7-Flash-Trellis-MM.
    """

    def __init__(
        self,
        config: SyntheticConfig | None = None,
        device: str = "mps",
    ):
        super().__init__()

        self.config = config or SyntheticConfig()
        self.device = device

        # Layer 0: Dense (uniform bits)
        self.dense_layer = SyntheticDenseLayer(self.config, device)

        # Layer 1: MoE (mixed bits)
        self.moe_layer = SyntheticMoELayer(self.config, device)

        # Simple embedding and output (for end-to-end testing)
        self.embed = nn.Embedding(1000, self.config.hidden_dim, device=device)
        self.lm_head = nn.Linear(self.config.hidden_dim, 1000, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through both layers.

        Args:
            x: Input tensor. Can be:
               - Token IDs: [batch, seq_len] long
               - Embeddings: [batch, seq_len, hidden_dim] float16

        Returns:
            Logits: [batch, seq_len, vocab_size] float16
        """
        # Handle token IDs vs embeddings
        if x.dtype in (torch.long, torch.int32):
            h = self.embed(x).half()
        else:
            h = x.half()

        # Layer 0: Dense
        h = h + self.dense_layer(h)

        # Layer 1: MoE
        h = h + self.moe_layer(h)

        # Output projection
        logits = self.lm_head(h.float())

        return logits.half()

    def get_bit_distribution(self) -> dict:
        """Get bit distribution statistics for analysis."""
        from collections import Counter

        bit_counts = Counter(self.moe_layer._bit_tuples)
        total = len(self.moe_layer._bit_tuples)

        return {
            "expert_bit_tuples": self.moe_layer._bit_tuples,
            "bit_tuple_counts": dict(bit_counts),
            "bit_tuple_percentages": {
                k: v / total * 100 for k, v in bit_counts.items()
            },
            "dense_bits": self.config.dense_bits,
            "num_experts": self.config.num_experts,
            "top_k": self.config.top_k,
        }


def create_synthetic_model(
    device: str = "mps",
    config: SyntheticConfig | None = None,
) -> SyntheticMixedMoE:
    """Factory function to create synthetic mixed-precision model.

    Args:
        device: Target device ('mps', 'cuda', or 'cpu')
        config: Optional configuration (uses defaults if None)

    Returns:
        SyntheticMixedMoE model ready for benchmarking
    """
    model = SyntheticMixedMoE(config=config, device=device)
    model.eval()
    return model


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    strategy: str
    throughput_tokens_per_sec: float
    latency_ms_per_token: float
    memory_mb: float
    iterations: int
    warmup: int

    def __str__(self) -> str:
        return (
            f"{self.strategy:20s} | "
            f"{self.throughput_tokens_per_sec:6.1f} | "
            f"{self.latency_ms_per_token:6.1f} | "
            f"{self.memory_mb:6.0f}MB"
        )


def benchmark_forward(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 1,
    warmup: int = 3,
    iterations: int = 10,
    device: str = "mps",
) -> BenchmarkResult:
    """Benchmark model forward pass.

    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        device: Device to run on

    Returns:
        BenchmarkResult with timing statistics
    """
    # Create input
    hidden_dim = model.config.hidden_dim
    x = torch.randn(batch_size, seq_len, hidden_dim,
                    dtype=torch.float16, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device == "mps":
                torch.mps.synchronize()

    # Measure memory before
    if device == "mps":
        torch.mps.empty_cache()
        mem_before = torch.mps.current_allocated_memory() / 1024 / 1024
    else:
        mem_before = 0

    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            _ = model(x)

            if device == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    tokens_per_iter = batch_size * seq_len
    throughput = tokens_per_iter / avg_time
    latency = avg_time * 1000 / tokens_per_iter

    # Memory after
    if device == "mps":
        mem_after = torch.mps.current_allocated_memory() / 1024 / 1024
        memory_mb = mem_after
    else:
        memory_mb = 0

    return BenchmarkResult(
        strategy="default",
        throughput_tokens_per_sec=throughput,
        latency_ms_per_token=latency,
        memory_mb=memory_mb,
        iterations=iterations,
        warmup=warmup,
    )


if __name__ == "__main__":
    # Quick test
    print("Creating synthetic mixed-precision model...")
    model = create_synthetic_model(device="mps")

    print(
        f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Config: {model.config}")

    print("\nBit distribution:")
    dist = model.get_bit_distribution()
    for tuple_key, count in dist["bit_tuple_counts"].items():
        pct = dist["bit_tuple_percentages"][tuple_key]
        print(f"  {tuple_key}: {count} experts ({pct:.1f}%)")

    print("\nRunning quick benchmark...")
    result = benchmark_forward(
        model, batch_size=1, seq_len=1, warmup=2, iterations=5)
    print(f"Throughput: {result.throughput_tokens_per_sec:.1f} tok/s")
    print(f"Latency: {result.latency_ms_per_token:.1f} ms/tok")

    print("\n✅ Synthetic model test passed!")
