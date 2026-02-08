#!/usr/bin/env python3
"""
Benchmark different attention implementations.

Tests:
- Standard attention (naive O(N^2) memory)
- Flash attention (tiled, memory-efficient)
- Fused QKV (fused projections)
- GQA (Grouped Query Attention)

Metrics:
- Time (latency, throughput)
- Memory usage
- Accuracy (vs reference implementation)
"""

from __future__ import annotations

import argparse
import sys
import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

# Add contrib paths for book-maker implementations
_ROOT = Path(__file__).resolve().parents[1]
_BOOK_MAKER_PATH = (
    _ROOT
    / "contrib"
    / "book-maker"
    / "books"
    / "physics-of-llm-inference"
    / "code"
    / "ch01"
)
sys.path.insert(0, str(_BOOK_MAKER_PATH))
sys.path.insert(0, str(_BOOK_MAKER_PATH.parent / "ch06"))


def ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for this benchmark")


@dataclass
class AttentionConfig:
    """Configuration for attention benchmark."""

    batch_size: int
    num_heads: int
    num_kv_heads: int | None  # None for standard MHA, different for GQA
    seq_len: int
    head_dim: int
    causal: bool = True

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    def __str__(self) -> str:
        gqa_str = f" (GQA: {self.num_kv_heads} KV heads)" if self.num_kv_heads else ""
        return f"B={self.batch_size}, H={self.num_heads}, N={self.seq_len}, D={self.head_dim}{gqa_str}"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    implementation: str
    config: AttentionConfig
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_mb: float
    max_error: float  # vs reference
    throughput_tokens_per_sec: float
    tflops: float

    def to_dict(self) -> dict:
        return {
            "implementation": self.implementation,
            "batch_size": self.config.batch_size,
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "seq_len": self.config.seq_len,
            "head_dim": self.config.head_dim,
            "causal": self.config.causal,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "memory_mb": self.memory_mb,
            "max_error": self.max_error,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "tflops": self.tflops,
        }


# =============================================================================
# Attention Implementations
# =============================================================================


def standard_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Standard naive attention: O(N^2) memory for attention matrix."""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


def flash_attention_tiled(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """
    Tiled flash attention implementation.
    Uses online softmax to avoid materializing full attention matrix.
    """
    batch, heads, seq_len, head_dim = q.shape
    scale = head_dim ** -0.5

    # Tile sizes
    BLOCK_Q = 64
    BLOCK_K = 64

    output = torch.zeros_like(q)
    row_max = torch.full((batch, heads, seq_len), float("-inf"), device=q.device, dtype=q.dtype)
    row_sum = torch.zeros((batch, heads, seq_len), device=q.device, dtype=q.dtype)

    num_q_blocks = (seq_len + BLOCK_Q - 1) // BLOCK_Q
    num_k_blocks = (seq_len + BLOCK_K - 1) // BLOCK_K

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * BLOCK_Q
        q_end = min(q_start + BLOCK_Q, seq_len)
        q_block = q[:, :, q_start:q_end, :]

        block_output = torch.zeros_like(q_block)
        block_max = torch.full(
            (batch, heads, q_end - q_start), float("-inf"), device=q.device, dtype=q.dtype
        )
        block_sum = torch.zeros((batch, heads, q_end - q_start), device=q.device, dtype=q.dtype)

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * BLOCK_K
            k_end = min(k_start + BLOCK_K, seq_len)

            # Causal masking: skip if q block is entirely after k block
            if causal and k_start > q_end:
                continue

            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            new_max = torch.maximum(block_max, scores.max(dim=-1).values)

            scale_old = torch.exp(block_max - new_max)
            scale_new = torch.exp(scores - new_max.unsqueeze(-1))

            new_sum = block_sum * scale_old + scale_new.sum(dim=-1)

            block_output = (
                block_output * block_sum.unsqueeze(-1) * scale_old.unsqueeze(-1)
                + torch.matmul(scale_new, v_block)
            ) / new_sum.unsqueeze(-1)

            block_max = new_max
            block_sum = new_sum

        output[:, :, q_start:q_end, :] = block_output
        row_max[:, :, q_start:q_end] = block_max
        row_sum[:, :, q_start:q_end] = block_sum

    return output


class FusedQKVAttention(nn.Module):
    """
    Multi-head attention with fused QKV projection.
    Single matrix multiply for Q, K, V instead of three separate ones.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection: single matmul for all three
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # [batch, seq, 3 * hidden]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        return output


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Uses fewer KV heads than Q heads to reduce memory and computation.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat K, V for GQA
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        return output


class StandardMultiHeadAttention(nn.Module):
    """Standard MHA with separate Q, K, V projections."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        return output


# =============================================================================
# Benchmark Functions
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def measure_memory(func: Callable, config: AttentionConfig, *args, **kwargs) -> tuple[torch.Tensor, float]:
    """Run function and measure peak memory usage."""
    device = get_device()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        result = func(*args, **kwargs)

        torch.cuda.synchronize(device)
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return result, memory_mb
    else:
        # For non-CUDA devices, estimate memory from tensor sizes
        result = func(*args, **kwargs)
        # Rough estimate: Q + K + V + output + intermediate
        # Input: [batch, seq, hidden], intermediate: attention scores [batch, heads, seq, seq]
        batch = config.batch_size
        seq = config.seq_len
        heads = config.num_heads
        dim = config.head_dim
        # QKV projections + attention matrix + output
        memory_bytes = (
            3 * batch * seq * heads * dim * 2 +  # Q, K, V in FP16
            batch * heads * seq * seq * 4 +       # Attention scores in FP32
            batch * seq * heads * dim * 2         # Output in FP16
        )
        memory_mb = memory_bytes / (1024 * 1024)
        return result, memory_mb


def calculate_tflops(config: AttentionConfig, time_ms: float) -> float:
    """Calculate TFLOPS for attention computation."""
    # Attention FLOPs:
    # Q @ K^T: 2 * B * H * N^2 * D
    # Softmax: 4 * B * H * N^2 (approx)
    # Attn @ V: 2 * B * H * N^2 * D
    # Total: ~4 * B * H * N^2 * D

    flops = 4 * config.batch_size * config.num_heads * config.seq_len * config.seq_len * config.head_dim

    if config.num_kv_heads and config.num_kv_heads < config.num_heads:
        # GQA: K, V projection is cheaper
        kv_factor = config.num_kv_heads / config.num_heads
        # Adjust for cheaper KV operations (roughly 1/3 of attention)
        flops = flops * (2 / 3 + kv_factor / 3)

    tflops = (flops / (time_ms / 1000)) / 1e12
    return tflops


def benchmark_implementation(
    name: str,
    config: AttentionConfig,
    forward_fn: Callable,
    warmup: int = 5,
    iterations: int = 20,
    reference_output: torch.Tensor | None = None,
) -> BenchmarkResult:
    """Benchmark a single attention implementation."""
    ensure_torch()
    device = get_device()

    # Create input tensors
    x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim, device=device)

    # Warmup
    for _ in range(warmup):
        _ = forward_fn(x)

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = forward_fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    times = np.array(times)

    # Measure memory
    _, memory_mb = measure_memory(forward_fn, config, x)

    # Calculate accuracy vs reference
    max_error = 0.0
    if reference_output is not None:
        with torch.no_grad():
            output = forward_fn(x)
            max_error = (output - reference_output).abs().max().item()

    # Calculate throughput
    total_tokens = config.batch_size * config.seq_len
    throughput = total_tokens / (np.mean(times) / 1000)  # tokens/sec

    tflops = calculate_tflops(config, np.mean(times))

    return BenchmarkResult(
        implementation=name,
        config=config,
        mean_time_ms=float(np.mean(times)),
        std_time_ms=float(np.std(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        memory_mb=memory_mb,
        max_error=max_error,
        throughput_tokens_per_sec=throughput,
        tflops=tflops,
    )


def benchmark_standard_attention(config: AttentionConfig, **kwargs) -> BenchmarkResult:
    """Benchmark standard attention with separate projections."""
    ensure_torch()
    device = get_device()

    model = StandardMultiHeadAttention(config.hidden_dim, config.num_heads).to(device)
    model.eval()

    def forward_fn(x):
        with torch.no_grad():
            return model(x, causal=config.causal)

    return benchmark_implementation("Standard Attention", config, forward_fn, **kwargs)


def benchmark_fused_qkv(config: AttentionConfig, **kwargs) -> BenchmarkResult:
    """Benchmark fused QKV attention."""
    ensure_torch()
    device = get_device()

    model = FusedQKVAttention(config.hidden_dim, config.num_heads).to(device)
    model.eval()

    def forward_fn(x):
        with torch.no_grad():
            return model(x, causal=config.causal)

    return benchmark_implementation("Fused QKV", config, forward_fn, **kwargs)


def benchmark_gqa(config: AttentionConfig, **kwargs) -> BenchmarkResult:
    """Benchmark Grouped Query Attention."""
    ensure_torch()
    device = get_device()

    if config.num_kv_heads is None:
        # Default to 4:1 ratio for GQA
        config.num_kv_heads = max(1, config.num_heads // 4)

    model = GroupedQueryAttention(
        config.hidden_dim, config.num_heads, config.num_kv_heads
    ).to(device)
    model.eval()

    def forward_fn(x):
        with torch.no_grad():
            return model(x, causal=config.causal)

    return benchmark_implementation(
        f"GQA ({config.num_kv_heads} KV heads)", config, forward_fn, **kwargs
    )


def benchmark_flash_attention(config: AttentionConfig, **kwargs) -> BenchmarkResult:
    """Benchmark flash attention using optimized SDPA when available."""
    ensure_torch()
    device = get_device()

    model = StandardMultiHeadAttention(config.hidden_dim, config.num_heads).to(device)
    model.eval()

    # Check if optimized SDPA is available
    has_sdpa = hasattr(F, 'scaled_dot_product_attention')

    def flash_forward(x):
        with torch.no_grad():
            batch, seq_len, _ = x.shape

            q = model.q_proj(x)
            k = model.k_proj(x)
            v = model.v_proj(x)

            q = q.view(batch, seq_len, model.num_heads, model.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, model.num_heads, model.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, model.num_heads, model.head_dim).transpose(1, 2)

            if has_sdpa:
                # Use PyTorch's optimized SDPA (includes Flash Attention kernels)
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, is_causal=config.causal
                )
            else:
                # Fall back to tiled implementation
                attn_output = flash_attention_tiled(q, k, v, causal=config.causal)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            return model.o_proj(attn_output)

    impl_name = "Flash Attention (SDPA)" if has_sdpa else "Flash Attention (tiled)"
    return benchmark_implementation(impl_name, config, flash_forward, **kwargs)


def generate_reference_output(config: AttentionConfig) -> torch.Tensor:
    """Generate reference output using double precision."""
    ensure_torch()
    device = get_device()

    model = StandardMultiHeadAttention(config.hidden_dim, config.num_heads).to(device)
    model.eval()

    x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim, device=device)

    # Compute in float64 for reference
    model = model.double()
    x = x.double()

    with torch.no_grad():
        return model(x, causal=config.causal).float()


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 120)
    print(
        f"{'Implementation':<30} {'Time (ms)':<15} {'Memory (MB)':<15} {'Throughput (tok/s)':<20} {'TFLOPS':<10} {'Max Error':<12}"
    )
    print("-" * 120)

    for r in results:
        time_str = f"{r.mean_time_ms:.2f} ± {r.std_time_ms:.2f}"
        print(
            f"{r.implementation:<30} {time_str:<15} {r.memory_mb:<15.1f} {r.throughput_tokens_per_sec:<20.1f} {r.tflops:<10.2f} {r.max_error:<12.6f}"
        )

    print("=" * 120)


def print_comparison(results: list[BenchmarkResult], baseline: str = "Standard Attention") -> None:
    """Print speedup comparison relative to baseline."""
    baseline_result = next((r for r in results if r.implementation == baseline), None)
    if not baseline_result:
        return

    print(f"\nSpeedup vs {baseline}:")
    print("-" * 50)
    for r in results:
        speedup = baseline_result.mean_time_ms / r.mean_time_ms
        memory_ratio = r.memory_mb / baseline_result.memory_mb
        print(f"  {r.implementation:<30} {speedup:>6.2f}x speed, {memory_ratio:>6.2f}x memory")


def run_benchmark_suite(
    configs: list[AttentionConfig],
    implementations: list[str],
    warmup: int = 5,
    iterations: int = 20,
) -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    ensure_torch()

    all_results = []

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config}")
        print(f"{'=' * 80}")

        # Generate reference output for accuracy comparison
        reference = None
        if "Standard Attention" in implementations:
            print("Generating reference output...")
            reference = generate_reference_output(config)

        results = []

        if "standard" in implementations:
            print("Benchmarking Standard Attention...")
            results.append(benchmark_standard_attention(config, warmup=warmup, iterations=iterations))

        if "fused" in implementations:
            print("Benchmarking Fused QKV...")
            results.append(
                benchmark_fused_qkv(
                    config, warmup=warmup, iterations=iterations, reference_output=reference
                )
            )

        if "gqa" in implementations:
            print("Benchmarking GQA...")
            results.append(
                benchmark_gqa(
                    config, warmup=warmup, iterations=iterations, reference_output=reference
                )
            )

        if "flash" in implementations:
            print("Benchmarking Flash Attention...")
            results.append(
                benchmark_flash_attention(
                    config, warmup=warmup, iterations=iterations, reference_output=reference
                )
            )

        print_results(results)
        if len(results) > 1:
            baseline = results[0].implementation if results else "Standard Attention"
            print_comparison(results, baseline)

        all_results.extend(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention implementations")
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["standard", "fused", "gqa", "flash", "all"],
        default=["all"],
        help="Which implementations to benchmark",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=2, help="For GQA")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for benchmarking")
        sys.exit(1)

    # Determine which implementations to run
    implementations = args.implementations
    if "all" in implementations:
        implementations = ["standard", "fused", "gqa", "flash"]

    # Create configurations
    configs = []
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            configs.append(
                AttentionConfig(
                    batch_size=batch_size,
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    seq_len=seq_len,
                    head_dim=args.head_dim,
                    causal=args.causal,
                )
            )

    print(f"\n{'#' * 80}")
    print("# Attention Implementation Benchmark")
    print(f"# Device: {get_device()}")
    print(f"# Implementations: {', '.join(implementations)}")
    print(f"# Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"{'#' * 80}")

    results = run_benchmark_suite(
        configs=configs,
        implementations=implementations,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # Save results if requested
    if args.output:
        import json
        output_data = [r.to_dict() for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
