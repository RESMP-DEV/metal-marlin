#!/usr/bin/env python3
"""
A/B Testing Framework for Optimized Kernel Implementations.

This module provides statistical A/B testing for kernel optimizations:
- Create variant implementations with specific optimizations
- Benchmark against baseline with statistical rigor
- Use t-tests to validate improvements are significant
- Keep winners, discard losers

Example:
    tester = KernelABTester()
    
    # Register baseline and variant
    tester.register_baseline("attention_standard", standard_attention_fn)
    tester.register_variant("attention_fused", fused_attention_fn, "Fused QKV projection")
    
    # Run A/B test
    result = tester.run_ab_test(
        "attention_fused",
        config=attention_config,
        iterations=100,
        confidence_level=0.99
    )
    
    # Apply winning implementation
    if result.winner == "variant":
        tester.apply_winner("attention_fused")
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy import stats

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

T = TypeVar("T")


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""

    iterations: int = 50
    warmup: int = 10
    confidence_level: float = 0.95
    min_speedup: float = 1.05  # Minimum 5% improvement to consider
    device: str = "cuda"


@dataclass
class TimingResult:
    """Timing results from a single benchmark run."""

    name: str
    times_ms: list[float] = field(default_factory=list)
    memory_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms, ddof=1))

    @property
    def min_ms(self) -> float:
        return float(np.min(self.times_ms))

    @property
    def max_ms(self) -> float:
        return float(np.max(self.times_ms))

    @property
    def median_ms(self) -> float:
        return float(np.median(self.times_ms))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.times_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.times_ms, 99))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "memory_mb": self.memory_mb,
            "samples": len(self.times_ms),
            "metadata": self.metadata,
        }


@dataclass
class ABTestResult:
    """Result of an A/B test between baseline and variant."""

    test_name: str
    baseline: TimingResult
    variant: TimingResult
    speedup: float
    p_value: float
    confidence_level: float
    is_significant: bool
    winner: str  # "baseline", "variant", or "tie"
    recommendation: str
    effect_size: float  # Cohen's d
    statistical_power: float

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "baseline": self.baseline.to_dict(),
            "variant": self.variant.to_dict(),
            "speedup": self.speedup,
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "is_significant": self.is_significant,
            "winner": self.winner,
            "recommendation": self.recommendation,
            "effect_size": self.effect_size,
            "statistical_power": self.statistical_power,
        }

    def __str__(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"A/B Test Result: {self.test_name}",
            f"{'='*70}",
            f"Baseline: {self.baseline.name}",
            f"  Mean: {self.baseline.mean_ms:.4f} ± {self.baseline.std_ms:.4f} ms",
            f"  P95:  {self.baseline.p95_ms:.4f} ms",
            f"  P99:  {self.baseline.p99_ms:.4f} ms",
            "",
            f"Variant: {self.variant.name}",
            f"  Mean: {self.variant.mean_ms:.4f} ± {self.variant.std_ms:.4f} ms",
            f"  P95:  {self.variant.p95_ms:.4f} ms",
            f"  P99:  {self.variant.p99_ms:.4f} ms",
            "",
            f"Speedup: {self.speedup:.3f}x ({(self.speedup-1)*100:+.1f}%)",
            f"P-value: {self.p_value:.6f}",
            f"Effect size (Cohen's d): {self.effect_size:.3f}",
            f"Statistical power: {self.statistical_power:.3f}",
            "",
            f"Winner: {self.winner.upper()}",
            f"Statistically significant: {'YES' if self.is_significant else 'NO'}",
            "",
            f"Recommendation: {self.recommendation}",
            f"{'='*70}",
        ]
        return "\n".join(lines)


class KernelRegistry:
    """Registry for kernel implementations."""

    def __init__(self):
        self._kernels: dict[str, dict] = {}
        self._baselines: dict[str, str] = {}

    def register_baseline(
        self, name: str, fn: Callable, description: str = ""
    ) -> None:
        """Register a baseline implementation."""
        self._kernels[name] = {
            "fn": fn,
            "type": "baseline",
            "description": description or f"Baseline: {name}",
        }

    def register_variant(
        self,
        name: str,
        fn: Callable,
        baseline_name: str,
        description: str = "",
        optimization_notes: str = "",
    ) -> None:
        """Register a variant implementation to test against a baseline."""
        if baseline_name not in self._kernels:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        self._kernels[name] = {
            "fn": fn,
            "type": "variant",
            "baseline": baseline_name,
            "description": description or f"Variant: {name}",
            "optimization_notes": optimization_notes,
        }

    def get_kernel(self, name: str) -> Callable:
        """Get kernel function by name."""
        if name not in self._kernels:
            raise ValueError(f"Kernel '{name}' not found")
        return self._kernels[name]["fn"]

    def get_baseline(self, variant_name: str) -> str:
        """Get baseline name for a variant."""
        if variant_name not in self._kernels:
            raise ValueError(f"Variant '{variant_name}' not found")
        kernel = self._kernels[variant_name]
        if kernel["type"] != "variant":
            raise ValueError(f"'{variant_name}' is not a variant")
        return kernel["baseline"]

    def list_kernels(self) -> list[str]:
        """List all registered kernel names."""
        return list(self._kernels.keys())

    def list_baselines(self) -> list[str]:
        """List all baseline kernel names."""
        return [k for k, v in self._kernels.items() if v["type"] == "baseline"]

    def list_variants(self, baseline: str | None = None) -> list[str]:
        """List variant kernel names, optionally filtered by baseline."""
        variants = [k for k, v in self._kernels.items() if v["type"] == "variant"]
        if baseline:
            variants = [
                k for k in variants if self._kernels[k].get("baseline") == baseline
            ]
        return variants


class KernelABTester:
    """A/B testing framework for kernel implementations."""

    def __init__(self):
        self.registry = KernelRegistry()
        self.test_results: dict[str, ABTestResult] = {}
        self._active_implementations: dict[str, str] = {}  # baseline -> active variant

    def register_baseline(
        self, name: str, fn: Callable, description: str = ""
    ) -> None:
        """Register a baseline implementation."""
        self.registry.register_baseline(name, fn, description)
        self._active_implementations[name] = name

    def register_variant(
        self,
        name: str,
        fn: Callable,
        baseline_name: str,
        description: str = "",
        optimization_notes: str = "",
    ) -> None:
        """Register a variant implementation to test."""
        self.registry.register_variant(
            name, fn, baseline_name, description, optimization_notes
        )

    def benchmark_kernel(
        self,
        name: str,
        kernel_fn: Callable,
        input_fn: Callable[[], T],
        iterations: int = 50,
        warmup: int = 10,
        device: str = "cuda",
    ) -> TimingResult:
        """Benchmark a single kernel implementation."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for benchmarking")

        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

        # Prepare inputs
        inputs = input_fn()

        # Warmup
        for _ in range(warmup):
            _ = kernel_fn(inputs)

        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)

        # Timed iterations
        times = []
        for _ in range(iterations):
            # Regenerate inputs for each iteration to avoid cache effects
            inputs = input_fn()

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)

            start = time.perf_counter()
            _ = kernel_fn(inputs)

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        # Memory measurement (if CUDA)
        memory_mb = 0.0
        if device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
            inputs = input_fn()
            _ = kernel_fn(inputs)
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            memory_mb = torch.cuda.max_memory_allocated(device_obj) / (1024 * 1024)

        return TimingResult(name=name, times_ms=times, memory_mb=memory_mb)

    def run_ab_test(
        self,
        variant_name: str,
        input_fn: Callable[[], T],
        config: ABTestConfig | None = None,
    ) -> ABTestResult:
        """Run A/B test between baseline and variant.

        Args:
            variant_name: Name of variant to test
            input_fn: Function that generates input data
            config: Test configuration

        Returns:
            ABTestResult with statistical analysis
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for A/B testing")

        cfg = config or ABTestConfig()
        baseline_name = self.registry.get_baseline(variant_name)

        baseline_fn = self.registry.get_kernel(baseline_name)
        variant_fn = self.registry.get_kernel(variant_name)

        print(f"\nRunning A/B Test: {baseline_name} vs {variant_name}")
        print(f"Configuration: {cfg.iterations} iterations, {cfg.warmup} warmup")
        print(f"Confidence level: {cfg.confidence_level}")

        # Benchmark baseline
        print(f"\nBenchmarking baseline: {baseline_name}...")
        baseline_result = self.benchmark_kernel(
            baseline_name,
            baseline_fn,
            input_fn,
            iterations=cfg.iterations,
            warmup=cfg.warmup,
            device=cfg.device,
        )

        # Benchmark variant
        print(f"Benchmarking variant: {variant_name}...")
        variant_result = self.benchmark_kernel(
            variant_name,
            variant_fn,
            input_fn,
            iterations=cfg.iterations,
            warmup=cfg.warmup,
            device=cfg.device,
        )

        # Statistical analysis
        speedup = baseline_result.mean_ms / variant_result.mean_ms

        # Paired t-test
        baseline_times = np.array(baseline_result.times_ms)
        variant_times = np.array(variant_result.times_ms)

        # Two-sided t-test: H0: mean_baseline == mean_variant
        t_stat, p_value = stats.ttest_ind(baseline_times, variant_times, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (baseline_result.std_ms**2 + variant_result.std_ms**2) / 2
        )
        effect_size = (
            (baseline_result.mean_ms - variant_result.mean_ms) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        # Statistical power (1 - beta)
        # Using statsmodels would be better, but we can estimate
        alpha = 1 - cfg.confidence_level
        n = cfg.iterations
        # Approximate power calculation
        critical_t = stats.t.ppf(1 - alpha / 2, df=2 * n - 2)
        ncp = effect_size * np.sqrt(n / 2)  # non-centrality parameter
        power = 1 - stats.nct.cdf(critical_t, df=2 * n - 2, nc=ncp)

        # Determine winner
        is_faster = speedup > cfg.min_speedup
        is_significant = p_value < (1 - cfg.confidence_level)

        if is_faster and is_significant:
            winner = "variant"
            recommendation = f"ACCEPT variant '{variant_name}' - statistically significant {speedup:.3f}x speedup"
        elif speedup >= 1.0:
            winner = "tie"
            recommendation = f"KEEP baseline - variant is not significantly faster (p={p_value:.4f})"
        else:
            winner = "baseline"
            recommendation = f"REJECT variant '{variant_name}' - slower than baseline"

        result = ABTestResult(
            test_name=f"{baseline_name}_vs_{variant_name}",
            baseline=baseline_result,
            variant=variant_result,
            speedup=speedup,
            p_value=p_value,
            confidence_level=cfg.confidence_level,
            is_significant=is_significant and is_faster,
            winner=winner,
            recommendation=recommendation,
            effect_size=effect_size,
            statistical_power=power,
        )

        self.test_results[f"{baseline_name}_vs_{variant_name}"] = result
        return result

    def run_batch_tests(
        self,
        tests: list[tuple[str, Callable]],
        input_fn: Callable[[str], T],
        config: ABTestConfig | None = None,
    ) -> list[ABTestResult]:
        """Run multiple A/B tests in sequence.

        Args:
            tests: List of (variant_name, input_generator) tuples
            input_fn: Function that takes variant_name and returns input generator
            config: Test configuration

        Returns:
            List of ABTestResult
        """
        results = []
        for variant_name in tests:
            if isinstance(variant_name, tuple):
                variant_name, custom_input_fn = variant_name
            else:
                custom_input_fn = lambda vn=variant_name: input_fn(vn)

            try:
                result = self.run_ab_test(variant_name, custom_input_fn, config)
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Error testing {variant_name}: {e}")

        return results

    def apply_winner(self, variant_name: str) -> None:
        """Mark a variant as the active implementation for its baseline."""
        baseline_name = self.registry.get_baseline(variant_name)
        self._active_implementations[baseline_name] = variant_name
        print(f"Applied: {variant_name} is now active for {baseline_name}")

    def get_active_implementation(self, baseline_name: str) -> str:
        """Get the currently active implementation for a baseline."""
        return self._active_implementations.get(baseline_name, baseline_name)

    def generate_report(self, output_path: str | None = None) -> str:
        """Generate a comprehensive test report."""
        lines = [
            "# A/B Test Report",
            "",
            "## Summary",
            f"Total tests: {len(self.test_results)}",
            f"Winners: {sum(1 for r in self.test_results.values() if r.winner == 'variant')}",
            f"Rejected: {sum(1 for r in self.test_results.values() if r.winner == 'baseline')}",
            f"Ties: {sum(1 for r in self.test_results.values() if r.winner == 'tie')}",
            "",
            "## Detailed Results",
            "",
        ]

        for test_name, result in self.test_results.items():
            lines.extend(
                [
                    f"### {test_name}",
                    f"- Winner: {result.winner}",
                    f"- Speedup: {result.speedup:.3f}x",
                    f"- P-value: {result.p_value:.6f}",
                    f"- Significant: {'Yes' if result.is_significant else 'No'}",
                    f"- Effect size: {result.effect_size:.3f}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Active Implementations",
                "",
                "| Baseline | Active Implementation |",
                "|----------|----------------------|",
            ]
        )

        for baseline, active in self._active_implementations.items():
            lines.append(f"| {baseline} | {active} |")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report)
            print(f"Report saved to {output_path}")

        return report

    def save_results(self, output_path: str) -> None:
        """Save all test results to JSON."""
        data = {
            "tests": {k: v.to_dict() for k, v in self.test_results.items()},
            "active_implementations": self._active_implementations,
        }
        Path(output_path).write_text(json.dumps(data, indent=2))
        print(f"Results saved to {output_path}")


# =============================================================================
# Example Kernel Implementations for Testing
# =============================================================================


def create_attention_kernels(tester: KernelABTester) -> None:
    """Register attention kernel variants for A/B testing."""
    if not TORCH_AVAILABLE:
        return

    # Baseline: Standard attention
    def standard_attention(hidden_state: torch.Tensor) -> torch.Tensor:
        """Standard MHA with separate projections."""
        batch, seq_len, hidden_dim = hidden_state.shape
        num_heads = 8
        head_dim = hidden_dim // num_heads

        # Separate projections
        q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)

        q = q_proj(hidden_state)
        k = k_proj(hidden_state)
        v = v_proj(hidden_state)

        # Reshape
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        # Attention
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        return o_proj(out)

    # Variant: Fused QKV
    def fused_qkv_attention(hidden_state: torch.Tensor) -> torch.Tensor:
        """Fused QKV projection attention."""
        batch, seq_len, hidden_dim = hidden_state.shape
        num_heads = 8
        head_dim = hidden_dim // num_heads

        # Fused projection
        qkv_proj = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=False).to(hidden_state.device)
        o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)

        qkv = qkv_proj(hidden_state)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        # Attention
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        return o_proj(out)

    # Variant: SDPA optimized
    def sdpa_attention(hidden_state: torch.Tensor) -> torch.Tensor:
        """Attention using PyTorch's optimized SDPA."""
        batch, seq_len, hidden_dim = hidden_state.shape
        num_heads = 8
        head_dim = hidden_dim // num_heads

        q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)

        q = q_proj(hidden_state)
        k = k_proj(hidden_state)
        v = v_proj(hidden_state)

        # Reshape
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        # Optimized SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        return o_proj(out)

    tester.register_baseline(
        "attention_standard",
        standard_attention,
        "Standard attention with separate Q/K/V projections",
    )

    tester.register_variant(
        "attention_fused",
        fused_qkv_attention,
        "attention_standard",
        "Fused QKV projection",
        "Single matmul for Q/K/V reduces kernel launches",
    )

    tester.register_variant(
        "attention_sdpa",
        sdpa_attention,
        "attention_standard",
        "PyTorch SDPA",
        "Uses optimized Flash Attention kernels when available",
    )


def create_kv_cache_kernels(tester: KernelABTester) -> None:
    """Register KV cache kernel variants for A/B testing."""
    if not TORCH_AVAILABLE:
        return

    # Baseline: BHSD layout
    def kvcache_bhsd(hidden_state: torch.Tensor) -> torch.Tensor:
        """KV cache with BHSD layout."""
        batch, seq_len, hidden_dim = hidden_state.shape
        num_heads = 8
        head_dim = hidden_dim // num_heads
        num_layers = 4

        k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)

        k = k_proj(hidden_state)
        v = v_proj(hidden_state)

        # BHSD layout: [batch, heads, seq, dim]
        k = k.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Simulate cache write
        cache_shape = (batch, num_heads, seq_len * 2, head_dim)
        k_cache = torch.zeros(cache_shape, device=hidden_state.device, dtype=k.dtype)
        v_cache = torch.zeros(cache_shape, device=hidden_state.device, dtype=v.dtype)

        k_cache[:, :, :seq_len, :].copy_(k)
        v_cache[:, :, :seq_len, :].copy_(v)

        # Simulate read (what attention needs)
        k_read = k_cache[:, :, :seq_len, :]
        v_read = v_cache[:, :, :seq_len, :]

        return k_read + v_read

    # Variant: BSHD layout
    def kvcache_bshd(hidden_state: torch.Tensor) -> torch.Tensor:
        """KV cache with BSHD layout."""
        batch, seq_len, hidden_dim = hidden_state.shape
        num_heads = 8
        head_dim = hidden_dim // num_heads

        k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)
        v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(hidden_state.device)

        k = k_proj(hidden_state)
        v = v_proj(hidden_state)

        # BSHD layout: [batch, seq, heads, dim]
        k = k.view(batch, seq_len, num_heads, head_dim)
        v = v.view(batch, seq_len, num_heads, head_dim)

        # Simulate cache write (coalesced)
        cache_shape = (batch, seq_len * 2, num_heads, head_dim)
        k_cache = torch.zeros(cache_shape, device=hidden_state.device, dtype=k.dtype)
        v_cache = torch.zeros(cache_shape, device=hidden_state.device, dtype=v.dtype)

        k_cache[:, :seq_len, :, :].copy_(k)
        v_cache[:, :seq_len, :, :].copy_(v)

        # Simulate read (transpose back to BHSD)
        k_read = k_cache[:, :seq_len, :, :].permute(0, 2, 1, 3)
        v_read = v_cache[:, :seq_len, :, :].permute(0, 2, 1, 3)

        return k_read + v_read

    tester.register_baseline(
        "kvcache_bhsd",
        kvcache_bhsd,
        "KV cache with BHSD layout [batch, heads, seq, dim]",
    )

    tester.register_variant(
        "kvcache_bshd",
        kvcache_bshd,
        "kvcache_bhsd",
        "KV cache with BSHD layout [batch, seq, heads, dim]",
        "Better memory coalescing for single-token writes during decode",
    )


def run_example_ab_tests():
    """Run example A/B tests."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping example tests")
        return

    tester = KernelABTester()

    # Register kernels
    create_attention_kernels(tester)
    create_kv_cache_kernels(tester)

    # Configuration
    config = ABTestConfig(
        iterations=100,
        warmup=20,
        confidence_level=0.99,
        min_speedup=1.05,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running A/B tests on device: {device}")

    # Test attention variants
    def attention_input():
        return torch.randn(4, 512, 512, device=device)

    print("\n" + "=" * 70)
    print("ATTENTION KERNEL A/B TESTS")
    print("=" * 70)

    for variant in ["attention_fused", "attention_sdpa"]:
        try:
            result = tester.run_ab_test(variant, attention_input, config)
            print(result)

            if result.winner == "variant":
                tester.apply_winner(variant)
        except Exception as e:
            print(f"Error testing {variant}: {e}")

    # Test KV cache variants
    def kvcache_input():
        return torch.randn(1, 1, 512, device=device)  # Single token decode

    print("\n" + "=" * 70)
    print("KV CACHE A/B TESTS")
    print("=" * 70)

    try:
        result = tester.run_ab_test("kvcache_bshd", kvcache_input, config)
        print(result)

        if result.winner == "variant":
            tester.apply_winner("kvcache_bshd")
    except Exception as e:
        print(f"Error testing kvcache_bshd: {e}")

    # Generate report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = tester.generate_report()
    print(report)

    # Save results
    tester.save_results("benchmarks/results/ab_test_results.json")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="A/B Test Kernel Implementations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level for statistical test",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.05,
        help="Minimum speedup to consider significant",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/ab_test_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--test",
        choices=["attention", "kvcache", "all"],
        default="all",
        help="Which tests to run",
    )

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for A/B testing")
        return 1

    tester = KernelABTester()

    # Register kernels based on test selection
    if args.test in ("attention", "all"):
        create_attention_kernels(tester)
    if args.test in ("kvcache", "all"):
        create_kv_cache_kernels(tester)

    config = ABTestConfig(
        iterations=args.iterations,
        warmup=args.warmup,
        confidence_level=args.confidence,
        min_speedup=args.min_speedup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running A/B tests on device: {device}")
    print(f"Configuration: {config}")

    # Run tests
    if args.test in ("attention", "all"):
        def attention_input():
            return torch.randn(4, 512, 512, device=device)

        print("\n" + "=" * 70)
        print("ATTENTION KERNEL A/B TESTS")
        print("=" * 70)

        for variant in ["attention_fused", "attention_sdpa"]:
            if variant in tester.registry.list_variants():
                try:
                    result = tester.run_ab_test(variant, attention_input, config)
                    print(result)
                    if result.winner == "variant":
                        tester.apply_winner(variant)
                except Exception as e:
                    print(f"Error: {e}")

    if args.test in ("kvcache", "all"):
        def kvcache_input():
            return torch.randn(1, 1, 512, device=device)

        print("\n" + "=" * 70)
        print("KV CACHE A/B TESTS")
        print("=" * 70)

        if "kvcache_bshd" in tester.registry.list_variants():
            try:
                result = tester.run_ab_test("kvcache_bshd", kvcache_input, config)
                print(result)
                if result.winner == "variant":
                    tester.apply_winner("kvcache_bshd")
            except Exception as e:
                print(f"Error: {e}")

    # Generate and save report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    tester.generate_report("benchmarks/results/ab_test_report.md")
    tester.save_results(args.output)

    return 0


if __name__ == "__main__":
    exit(main())
