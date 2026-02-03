#!/usr/bin/env python3
"""Benchmark comparing old (Python indexing) vs new (Metal kernel) MoE dispatch.

The Problem:
    On MPS, advanced indexing operations like expert_weights[topk_indices[:, k]]
    trigger CPU-GPU synchronization that can take 20+ seconds for large tensors.
    This is because PyTorch must materialize the indices on CPU to perform the gather.

The Solution:
    Use Metal kernels (expert_gather, expert_scatter_add) that keep indices on GPU
    and perform the gather/scatter entirely in hardware.

Expected Results:
    - Old dispatch (Python indexing): 10-100+ ms per call
    - New dispatch (Metal kernel): <1ms per call
    - Speedup: 100x+ for typical MoE configurations

Usage:
    cd contrib/metal_marlin
    uv run python benchmark_moe_dispatch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Add contrib path for metal_marlin imports
sys.path.insert(0, str(Path(__file__).parent))

import torch

if TYPE_CHECKING:
    pass


def benchmark_old_dispatch(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    num_warmup: int = 2,
    num_iterations: int = 10,
) -> tuple[float, torch.Tensor]:
    """Old approach: Python loop with advanced indexing (SLOW).

    This simulates the naive MoE dispatch pattern where we:
    1. Loop over each top-k selection
    2. Use advanced indexing to select expert weights
    3. Compute matmul for each expert

    The advanced indexing (expert_weights[expert_idx]) triggers MPS synchronization
    because indices must be materialized on CPU.

    Args:
        hidden_states: [batch, hidden_dim] input activations
        expert_weights: [num_experts, hidden_dim, out_dim] expert weight matrices
        topk_indices: [batch, top_k] selected expert indices per token
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        (mean_time_ms, output_tensor)
    """
    batch_size, hidden_dim = hidden_states.shape
    _, _, out_dim = expert_weights.shape
    top_k = topk_indices.shape[1]

    # Warmup
    for _ in range(num_warmup):
        outputs = []
        for k in range(top_k):
            # This is the slow operation! Advanced indexing triggers CPU sync
            expert_idx = topk_indices[:, k]
            # Gather weights for selected experts
            selected_weights = expert_weights[expert_idx]  # [batch, hidden, out]
            # Compute: output = hidden @ weights -> [batch, out]
            output = torch.bmm(
                hidden_states.unsqueeze(1),  # [batch, 1, hidden]
                selected_weights,  # [batch, hidden, out]
            ).squeeze(1)  # [batch, out]
            outputs.append(output)
        torch.mps.synchronize()

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        outputs = []
        for k in range(top_k):
            expert_idx = topk_indices[:, k]
            selected_weights = expert_weights[expert_idx]
            output = torch.bmm(
                hidden_states.unsqueeze(1),
                selected_weights,
            ).squeeze(1)
            outputs.append(output)

        # Stack outputs: [batch, top_k, out]
        result = torch.stack(outputs, dim=1)

        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times), result


def benchmark_new_dispatch_batched(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    num_warmup: int = 2,
    num_iterations: int = 10,
) -> tuple[float, torch.Tensor]:
    """New approach: Batched dispatch avoiding MPS indexing (FAST).

    Uses the BatchedExpertDispatch class which groups tokens by expert
    and processes them efficiently without advanced indexing.

    Args:
        hidden_states: [batch, hidden_dim] input activations
        expert_weights: [num_experts, hidden_dim, out_dim] expert weight matrices
        topk_indices: [batch, top_k] selected expert indices per token
        routing_weights: [batch, top_k] routing probabilities
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        (mean_time_ms, output_tensor)
    """
    from metal_marlin.moe.batched_dispatch import batched_expert_forward

    batch_size, hidden_dim = hidden_states.shape
    num_experts = expert_weights.shape[0]

    def expert_fn(inputs: torch.Tensor, expert_id: int) -> torch.Tensor:
        """Process tokens through a single expert."""
        return inputs @ expert_weights[expert_id]

    # Warmup
    for _ in range(num_warmup):
        _ = batched_expert_forward(
            hidden_states,
            topk_indices,
            routing_weights,
            expert_fn,
            num_experts,
        )
        torch.mps.synchronize()

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        result = batched_expert_forward(
            hidden_states,
            topk_indices,
            routing_weights,
            expert_fn,
            num_experts,
        )

        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times), result


def benchmark_metal_gather(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    num_warmup: int = 2,
    num_iterations: int = 10,
) -> tuple[float, torch.Tensor]:
    """Metal kernel gather: Direct GPU gather without CPU sync.

    Uses expert_gather from metal_marlin.expert_ops which dispatches
    to a Metal kernel that performs the gather entirely on GPU.

    Note: This benchmarks just the gather operation, not the full MoE forward.

    Args:
        hidden_states: [batch, hidden_dim] input activations
        expert_weights: [num_experts, hidden_dim, out_dim] expert weight matrices
        topk_indices: [batch, top_k] selected expert indices per token
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        (mean_time_ms, gathered_weights)
    """
    from metal_marlin.expert_ops import expert_gather

    # Warmup
    for _ in range(num_warmup):
        _ = expert_gather(expert_weights, topk_indices.int())
        torch.mps.synchronize()

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        result = expert_gather(expert_weights, topk_indices.int())

        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times), result


def benchmark_pure_indexing(
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    num_warmup: int = 2,
    num_iterations: int = 10,
) -> tuple[float, torch.Tensor]:
    """Benchmark pure advanced indexing without matmul.

    This isolates the indexing overhead to show that the slow part
    is the gather operation, not the matmul.

    Args:
        expert_weights: [num_experts, hidden_dim, out_dim] expert weight matrices
        topk_indices: [batch, top_k] selected expert indices per token
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        (mean_time_ms, gathered_weights)
    """
    # Warmup
    for _ in range(num_warmup):
        _ = expert_weights[topk_indices[:, 0]]
        torch.mps.synchronize()

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        # Just the indexing operation
        result = expert_weights[topk_indices[:, 0]]

        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times), result


def main():
    print("=" * 70)
    print("MoE Dispatch Benchmark: Old (Python indexing) vs New (Metal kernel)")
    print("=" * 70)

    # Check device
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available. This benchmark requires Apple Silicon.")
        sys.exit(1)

    device = torch.device("mps")
    dtype = torch.float16

    # Configuration matching typical MoE models (e.g., Mixtral, DeepSeek)
    configs = [
        # (batch_size, num_experts, hidden_dim, out_dim, top_k)
        (16, 8, 4096, 14336, 2, "Small (16 tokens, 8 experts)"),
        (64, 8, 4096, 14336, 2, "Medium (64 tokens, 8 experts)"),
        (128, 64, 4096, 14336, 2, "Large (128 tokens, 64 experts)"),
        (256, 64, 4096, 14336, 2, "XL (256 tokens, 64 experts)"),
    ]

    for batch_size, num_experts, hidden_dim, out_dim, top_k, name in configs:
        print(f"\n{'=' * 70}")
        print(f"Config: {name}")
        print(f"  batch={batch_size}, experts={num_experts}, "
              f"hidden={hidden_dim}, out={out_dim}, top_k={top_k}")
        print("=" * 70)

        # Create test tensors
        hidden_states = torch.randn(
            batch_size, hidden_dim, device=device, dtype=dtype
        )
        expert_weights = torch.randn(
            num_experts, hidden_dim, out_dim, device=device, dtype=dtype
        )
        topk_indices = torch.randint(
            0, num_experts, (batch_size, top_k), device=device, dtype=torch.int64
        )
        routing_weights = torch.softmax(
            torch.randn(batch_size, top_k, device=device, dtype=dtype), dim=-1
        )

        # Memory usage
        weights_mb = expert_weights.numel() * 2 / 1024 / 1024
        print(f"\n  Expert weights memory: {weights_mb:.1f} MB")

        # Benchmark 1: Pure indexing (isolate the gather overhead)
        print("\n  [1] Pure Advanced Indexing (expert_weights[indices]):")
        try:
            pure_idx_ms, _ = benchmark_pure_indexing(
                expert_weights, topk_indices, num_warmup=1, num_iterations=5
            )
            print(f"      Time: {pure_idx_ms:8.2f} ms")
        except Exception as e:
            print(f"      ERROR: {e}")
            pure_idx_ms = float("inf")

        # Benchmark 2: Old dispatch with loop
        print("\n  [2] Old Dispatch (Python loop + indexing + bmm):")
        try:
            old_ms, old_result = benchmark_old_dispatch(
                hidden_states, expert_weights, topk_indices,
                num_warmup=1, num_iterations=5
            )
            print(f"      Time: {old_ms:8.2f} ms")
        except Exception as e:
            print(f"      ERROR: {e}")
            old_ms = float("inf")

        # Benchmark 3: New batched dispatch
        print("\n  [3] New Dispatch (BatchedExpertDispatch):")
        try:
            new_ms, new_result = benchmark_new_dispatch_batched(
                hidden_states, expert_weights, topk_indices, routing_weights,
                num_warmup=1, num_iterations=5
            )
            print(f"      Time: {new_ms:8.2f} ms")
        except Exception as e:
            print(f"      ERROR: {e}")
            new_ms = float("inf")

        # Benchmark 4: Metal gather kernel only
        print("\n  [4] Metal Gather Kernel (expert_gather):")
        try:
            metal_ms, _ = benchmark_metal_gather(
                hidden_states, expert_weights, topk_indices,
                num_warmup=1, num_iterations=5
            )
            print(f"      Time: {metal_ms:8.2f} ms")
        except Exception as e:
            print(f"      ERROR: {e}")
            metal_ms = float("inf")

        # Summary
        print("\n  Summary:")
        print("  " + "-" * 50)

        if old_ms != float("inf") and new_ms != float("inf"):
            speedup = old_ms / new_ms
            print(f"  Old dispatch:     {old_ms:8.2f} ms")
            print(f"  New dispatch:     {new_ms:8.2f} ms")
            print(f"  Speedup:          {speedup:8.1f}x")

            if speedup >= 100:
                print("  \u2705 TARGET MET: 100x+ speedup achieved!")
            elif speedup >= 10:
                print("  \u26a0\ufe0f PARTIAL: 10x+ speedup, not quite 100x")
            else:
                print("  \u274c BELOW TARGET: Less than 10x speedup")

        if pure_idx_ms != float("inf") and metal_ms != float("inf"):
            gather_speedup = pure_idx_ms / metal_ms
            print(f"\n  Pure indexing:    {pure_idx_ms:8.2f} ms")
            print(f"  Metal gather:     {metal_ms:8.2f} ms")
            print(f"  Gather speedup:   {gather_speedup:8.1f}x")

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
