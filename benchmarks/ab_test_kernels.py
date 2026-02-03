#!/usr/bin/env python3
"""
A/B Testing Framework for Metal Kernel Optimizations.

Provides statistical validation of kernel performance improvements using:
- Welch's t-test for comparing means with unequal variances
- Effect size (Cohen's d) for practical significance
- Bootstrap confidence intervals for robust estimation
- Mann-Whitney U test as non-parametric alternative

Usage:
    python benchmarks/ab_test_kernels.py --kernel gemm_trellis
    python benchmarks/ab_test_kernels.py --kernel attention --iterations 200
    python benchmarks/ab_test_kernels.py --all --export results/ab_tests.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import torch

# Ensure metal_marlin is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.kernels import HAS_MPS


class KernelFn(Protocol):
    """Protocol for kernel benchmark functions."""

    def __call__(self) -> torch.Tensor: ...


@dataclass
class ABTestResult:
    """Result from an A/B kernel comparison test."""

    kernel_name: str
    variant_a: str
    variant_b: str

    # Timing statistics (ms)
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    min_a: float
    min_b: float

    # Statistical tests
    t_statistic: float
    p_value: float
    cohens_d: float

    # Mann-Whitney U (non-parametric)
    u_statistic: float
    u_p_value: float

    # Bootstrap CI for difference (B - A)
    ci_lower: float
    ci_upper: float
    ci_level: float

    # Decision
    winner: str  # "A", "B", or "tie"
    speedup: float  # B/A ratio (< 1 means B is faster)
    significant: bool  # p < alpha
    practical: bool  # |Cohen's d| > threshold

    # Metadata
    iterations: int
    warmup: int
    alpha: float
    effect_size_threshold: float

    # Optional throughput metrics
    tflops_a: float | None = None
    tflops_b: float | None = None


@dataclass
class KernelVariant:
    """Definition of a kernel variant for A/B testing."""

    name: str
    description: str
    create_fn: Callable[..., KernelFn]
    # Optional parameters for variant configuration
    params: dict[str, int] = field(default_factory=dict)


def mps_sync() -> None:
    """Synchronize MPS device."""
    torch.mps.synchronize()


def welchs_ttest(
    samples_a: Sequence[float], samples_b: Sequence[float]
) -> tuple[float, float]:
    """
    Welch's t-test for comparing two samples with unequal variances.

    Returns (t_statistic, p_value).
    """
    n_a, n_b = len(samples_a), len(samples_b)
    mean_a = statistics.mean(samples_a)
    mean_b = statistics.mean(samples_b)

    # Use sample variance (ddof=1)
    var_a = statistics.variance(samples_a) if n_a > 1 else 0.0
    var_b = statistics.variance(samples_b) if n_b > 1 else 0.0

    # Standard error of the difference
    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a + var_b) > 0 else 1e-10

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    if var_a > 0 and var_b > 0:
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else n_a + n_b - 2
    else:
        df = n_a + n_b - 2

    # Approximate p-value using normal distribution for large df
    # For proper t-distribution, would need scipy.stats.t.sf
    # Using normal approximation: p = 2 * (1 - Phi(|t|))
    # This is accurate for df > 30
    p_value = 2 * _normal_sf(abs(t_stat))

    return t_stat, p_value


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) for standard normal distribution."""
    # Using complementary error function approximation
    # erfc(x/sqrt(2))/2
    return 0.5 * math.erfc(x / math.sqrt(2))


def cohens_d(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
    """
    Cohen's d effect size for the difference between two groups.

    Uses pooled standard deviation.
    """
    n_a, n_b = len(samples_a), len(samples_b)
    mean_a = statistics.mean(samples_a)
    mean_b = statistics.mean(samples_b)

    var_a = statistics.variance(samples_a) if n_a > 1 else 0.0
    var_b = statistics.variance(samples_b) if n_b > 1 else 0.0

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10

    return (mean_a - mean_b) / pooled_std


def mann_whitney_u(
    samples_a: Sequence[float], samples_b: Sequence[float]
) -> tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Returns (U_statistic, p_value).
    """
    n_a, n_b = len(samples_a), len(samples_b)

    # Count how many times a value from A exceeds a value from B
    u_a = sum(1 for a in samples_a for b in samples_b if a > b)
    u_a += 0.5 * sum(1 for a in samples_a for b in samples_b if a == b)

    u_b = n_a * n_b - u_a

    # Use the smaller U for the test statistic
    u_stat = min(u_a, u_b)

    # Normal approximation for p-value (valid for n > 20)
    mean_u = n_a * n_b / 2
    std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    if std_u > 0:
        z = (u_stat - mean_u) / std_u
        p_value = 2 * _normal_sf(abs(z))
    else:
        p_value = 1.0

    return u_stat, p_value


def bootstrap_ci(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the difference in means (B - A).

    Uses percentile method.
    """
    import random

    differences: list[float] = []
    for _ in range(n_bootstrap):
        boot_a = random.choices(samples_a, k=len(samples_a))
        boot_b = random.choices(samples_b, k=len(samples_b))
        diff = statistics.mean(boot_b) - statistics.mean(boot_a)
        differences.append(diff)

    differences.sort()
    alpha = 1 - ci_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return differences[lower_idx], differences[upper_idx]


class ABTester:
    """
    A/B testing harness for kernel performance comparison.

    Runs both kernel variants with controlled conditions and applies
    statistical tests to determine if there's a significant difference.
    """

    def __init__(
        self,
        warmup: int = 50,
        iterations: int = 100,
        alpha: float = 0.05,
        effect_size_threshold: float = 0.2,  # Small effect by Cohen's conventions
        n_bootstrap: int = 10000,
        ci_level: float = 0.95,
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.alpha = alpha
        self.effect_size_threshold = effect_size_threshold
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.results: list[ABTestResult] = []

    def run(
        self,
        kernel_name: str,
        variant_a: KernelVariant,
        variant_b: KernelVariant,
        M: int,
        N: int,
        K: int,
    ) -> ABTestResult:
        """
        Run A/B test comparing two kernel variants.

        Interleaves A and B runs to reduce systematic bias from thermal
        throttling or background processes.
        """
        print(f"\n{'='*60}")
        print(f"A/B Test: {kernel_name}")
        print(f"  Variant A: {variant_a.name} - {variant_a.description}")
        print(f"  Variant B: {variant_b.name} - {variant_b.description}")
        print(f"  Problem size: M={M}, N={N}, K={K}")
        print(f"{'='*60}")

        # Create kernel functions
        fn_a = variant_a.create_fn(M, N, K, **variant_a.params)
        fn_b = variant_b.create_fn(M, N, K, **variant_b.params)

        # Warmup both variants
        print(f"Warming up ({self.warmup} iterations each)...")
        for _ in range(self.warmup):
            fn_a()
            mps_sync()
            fn_b()
            mps_sync()

        # Interleaved timing runs
        print(f"Running {self.iterations} interleaved iterations...")
        times_a: list[float] = []
        times_b: list[float] = []

        for i in range(self.iterations):
            # Alternate starting order to reduce bias
            if i % 2 == 0:
                # Run A first
                mps_sync()
                t0 = time.perf_counter()
                fn_a()
                mps_sync()
                times_a.append((time.perf_counter() - t0) * 1000)

                mps_sync()
                t0 = time.perf_counter()
                fn_b()
                mps_sync()
                times_b.append((time.perf_counter() - t0) * 1000)
            else:
                # Run B first
                mps_sync()
                t0 = time.perf_counter()
                fn_b()
                mps_sync()
                times_b.append((time.perf_counter() - t0) * 1000)

                mps_sync()
                t0 = time.perf_counter()
                fn_a()
                mps_sync()
                times_a.append((time.perf_counter() - t0) * 1000)

        # Statistical analysis
        print("Computing statistics...")
        mean_a = statistics.mean(times_a)
        mean_b = statistics.mean(times_b)
        std_a = statistics.stdev(times_a)
        std_b = statistics.stdev(times_b)

        t_stat, p_value = welchs_ttest(times_a, times_b)
        d = cohens_d(times_a, times_b)
        u_stat, u_p_value = mann_whitney_u(times_a, times_b)
        ci_lower, ci_upper = bootstrap_ci(
            times_a, times_b, self.n_bootstrap, self.ci_level
        )

        # TFLOPS calculation
        flops = 2.0 * M * N * K
        tflops_a = (flops / (mean_a / 1000)) / 1e12 if mean_a > 0 else None
        tflops_b = (flops / (mean_b / 1000)) / 1e12 if mean_b > 0 else None

        # Decision logic
        significant = p_value < self.alpha
        practical = abs(d) > self.effect_size_threshold
        speedup = mean_a / mean_b if mean_b > 0 else 1.0

        if significant and practical:
            # Statistically and practically significant
            if mean_b < mean_a:
                winner = "B"
            else:
                winner = "A"
        else:
            winner = "tie"

        result = ABTestResult(
            kernel_name=kernel_name,
            variant_a=variant_a.name,
            variant_b=variant_b.name,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            min_a=min(times_a),
            min_b=min(times_b),
            t_statistic=t_stat,
            p_value=p_value,
            cohens_d=d,
            u_statistic=u_stat,
            u_p_value=u_p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            winner=winner,
            speedup=speedup,
            significant=significant,
            practical=practical,
            iterations=self.iterations,
            warmup=self.warmup,
            alpha=self.alpha,
            effect_size_threshold=self.effect_size_threshold,
            tflops_a=tflops_a,
            tflops_b=tflops_b,
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def _print_result(self, r: ABTestResult) -> None:
        """Print formatted test result."""
        print("\nResults:")
        print(f"  Variant A ({r.variant_a}):")
        print(f"    Mean: {r.mean_a:.3f} ms ± {r.std_a:.3f} ms")
        print(f"    Min:  {r.min_a:.3f} ms")
        if r.tflops_a:
            print(f"    TFLOPS: {r.tflops_a:.2f}")

        print(f"  Variant B ({r.variant_b}):")
        print(f"    Mean: {r.mean_b:.3f} ms ± {r.std_b:.3f} ms")
        print(f"    Min:  {r.min_b:.3f} ms")
        if r.tflops_b:
            print(f"    TFLOPS: {r.tflops_b:.2f}")

        print("\nStatistical Tests:")
        print(f"  Welch's t-test: t={r.t_statistic:.3f}, p={r.p_value:.4f}")
        print(f"  Mann-Whitney U: U={r.u_statistic:.1f}, p={r.u_p_value:.4f}")
        print(f"  Cohen's d: {r.cohens_d:.3f}")
        print(
            f"  Bootstrap {r.ci_level*100:.0f}% CI for (B-A): [{r.ci_lower:.3f}, {r.ci_upper:.3f}] ms"
        )

        print("\nDecision:")
        if r.winner == "tie":
            print("  No significant difference detected")
            if not r.significant:
                print(f"    (p={r.p_value:.4f} >= α={r.alpha})")
            if not r.practical:
                print(
                    f"    (|d|={abs(r.cohens_d):.3f} < threshold={r.effect_size_threshold})"
                )
        else:
            speedup_pct = (r.speedup - 1) * 100 if r.speedup > 1 else (1 / r.speedup - 1) * 100
            faster = "B" if r.speedup > 1 else "A"
            print(f"  WINNER: {r.winner}")
            print(f"    {faster} is {speedup_pct:.1f}% faster (speedup: {r.speedup:.3f}x)")
            print(f"    p-value: {r.p_value:.4f} (significant at α={r.alpha})")
            print(f"    Effect size: {abs(r.cohens_d):.3f} (practical threshold: {r.effect_size_threshold})")

    def export_json(self, path: str | Path) -> None:
        """Export all results to JSON."""
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

    def print_summary(self) -> None:
        """Print summary of all A/B tests."""
        if not self.results:
            print("No test results.")
            return

        print("\n" + "=" * 70)
        print("A/B Test Summary")
        print("=" * 70)

        for r in self.results:
            status = "✓" if r.winner != "tie" else "~"
            if r.winner == "B":
                speedup = f"+{(r.speedup - 1) * 100:.1f}%"
            elif r.winner == "A":
                speedup = f"-{(1 / r.speedup - 1) * 100:.1f}%"
            else:
                speedup = "0%"

            print(
                f"  [{status}] {r.kernel_name}: {r.variant_a} vs {r.variant_b} → "
                f"Winner: {r.winner} ({speedup}, p={r.p_value:.3f})"
            )


# =============================================================================
# Kernel variant definitions
# =============================================================================


def create_gemm_baseline(M: int, N: int, K: int, **_) -> KernelFn:
    """Create baseline FP16 GEMM kernel function."""
    A = torch.randn(M, K, dtype=torch.float16, device="mps")
    B = torch.randn(K, N, dtype=torch.float16, device="mps")
    mps_sync()

    def fn() -> torch.Tensor:
        return A @ B

    return fn


def create_gemm_batched(M: int, N: int, K: int, batch_size: int = 4, **_) -> KernelFn:
    """Create batched GEMM for comparison."""
    # Simulate batched by doing multiple smaller GEMMs
    A = torch.randn(batch_size, M // batch_size, K, dtype=torch.float16, device="mps")
    B = torch.randn(K, N, dtype=torch.float16, device="mps")
    mps_sync()

    def fn() -> torch.Tensor:
        return torch.einsum("bik,kj->bij", A, B)

    return fn


def create_attention_baseline(
    M: int, N: int, K: int, num_heads: int = 8, **_
) -> KernelFn:
    """Create baseline attention kernel (scaled dot-product)."""
    seq_len = M
    head_dim = K // num_heads
    batch = 1

    Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    K_mat = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    scale = 1.0 / math.sqrt(head_dim)
    mps_sync()

    def fn() -> torch.Tensor:
        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, V)

    return fn


def create_attention_sdpa(M: int, N: int, K: int, num_heads: int = 8, **_) -> KernelFn:
    """Create attention using PyTorch's scaled_dot_product_attention."""
    seq_len = M
    head_dim = K // num_heads
    batch = 1

    Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    K_mat = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    mps_sync()

    def fn() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(Q, K_mat, V)

    return fn


def create_moe_baseline(
    M: int, N: int, K: int, num_experts: int = 8, top_k: int = 2, **_
) -> KernelFn:
    """Create baseline MoE dispatch (naive loop over experts)."""
    batch = M
    hidden = K
    out_dim = N

    activations = torch.randn(batch, hidden, dtype=torch.float16, device="mps")
    expert_weights = torch.randn(
        num_experts, hidden, out_dim, dtype=torch.float16, device="mps"
    )
    # Random routing
    expert_ids = torch.randint(0, num_experts, (batch, top_k), device="mps")
    expert_probs = torch.softmax(
        torch.randn(batch, top_k, dtype=torch.float16, device="mps"), dim=-1
    )
    mps_sync()

    def fn() -> torch.Tensor:
        output = torch.zeros(batch, out_dim, dtype=torch.float16, device="mps")
        for b in range(batch):
            for k in range(top_k):
                expert = expert_ids[b, k].item()
                prob = expert_probs[b, k]
                out = activations[b : b + 1] @ expert_weights[expert]
                output[b] += prob * out.squeeze(0)
        return output

    return fn


def create_moe_batched(
    M: int, N: int, K: int, num_experts: int = 8, top_k: int = 2, **_
) -> KernelFn:
    """Create batched MoE dispatch (group tokens by expert)."""
    batch = M
    hidden = K
    out_dim = N

    activations = torch.randn(batch, hidden, dtype=torch.float16, device="mps")
    expert_weights = torch.randn(
        num_experts, hidden, out_dim, dtype=torch.float16, device="mps"
    )
    expert_ids = torch.randint(0, num_experts, (batch, top_k), device="mps")
    expert_probs = torch.softmax(
        torch.randn(batch, top_k, dtype=torch.float16, device="mps"), dim=-1
    )
    mps_sync()

    def fn() -> torch.Tensor:
        output = torch.zeros(batch, out_dim, dtype=torch.float16, device="mps")

        # Group by expert
        for expert in range(num_experts):
            mask = (expert_ids == expert).any(dim=1)
            if not mask.any():
                continue

            # Batched GEMM for all tokens assigned to this expert
            batch_indices = torch.where(mask)[0]
            batch_acts = activations[batch_indices]
            batch_out = batch_acts @ expert_weights[expert]

            # Apply weights
            for i, b in enumerate(batch_indices.tolist()):
                for k in range(top_k):
                    if expert_ids[b, k] == expert:
                        output[b] += expert_probs[b, k] * batch_out[i]

        return output

    return fn


# =============================================================================
# Pre-defined kernel tests
# =============================================================================

KERNEL_TESTS: dict[str, tuple[KernelVariant, KernelVariant, tuple[int, int, int]]] = {
    "gemm_standard": (
        KernelVariant("standard", "Standard A @ B matmul", create_gemm_baseline),
        KernelVariant("batched", "Batched einsum matmul", create_gemm_batched, {"batch_size": 4}),
        (512, 4096, 4096),
    ),
    "attention_impl": (
        KernelVariant("manual", "Manual QK @ V attention", create_attention_baseline),
        KernelVariant("sdpa", "PyTorch SDPA", create_attention_sdpa),
        (128, 128, 1024),  # seq_len=128, head_dim=128 (8 heads)
    ),
    "moe_dispatch": (
        KernelVariant("naive", "Per-token loop dispatch", create_moe_baseline),
        KernelVariant("batched", "Expert-grouped batched", create_moe_batched),
        (64, 4096, 4096),  # 64 tokens, 4096 hidden, 4096 out
    ),
}


def main() -> None:
    """Main entry point for A/B kernel testing."""
    parser = argparse.ArgumentParser(
        description="A/B test kernel implementations with statistical validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--kernel",
        choices=list(KERNEL_TESTS.keys()),
        help="Specific kernel to test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all kernel tests",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.2,
        help="Minimum Cohen's d for practical significance (default: 0.2)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )
    args = parser.parse_args()

    if not HAS_MPS:
        print("ERROR: PyTorch MPS backend required for Metal kernel benchmarking.")
        sys.exit(1)

    if not args.kernel and not args.all:
        parser.print_help()
        print("\nAvailable kernel tests:")
        for name, (va, vb, size) in KERNEL_TESTS.items():
            print(f"  {name}: {va.name} vs {vb.name} @ {size}")
        sys.exit(0)

    tester = ABTester(
        warmup=args.warmup,
        iterations=args.iterations,
        alpha=args.alpha,
        effect_size_threshold=args.effect_threshold,
    )

    kernels_to_test = list(KERNEL_TESTS.keys()) if args.all else [args.kernel]

    for kernel_name in kernels_to_test:
        variant_a, variant_b, (M, N, K) = KERNEL_TESTS[kernel_name]
        tester.run(kernel_name, variant_a, variant_b, M, N, K)

    tester.print_summary()

    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        tester.export_json(export_path)
        print(f"\nResults exported to {export_path}")


if __name__ == "__main__":
    main()
