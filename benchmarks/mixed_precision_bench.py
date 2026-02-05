"""Benchmark harness for mixed-precision MoE optimization.

This module provides a comprehensive benchmark framework for comparing
different dispatch strategies for mixed-precision quantized models.

Usage:
    >>> from benchmarks.mixed_precision_bench import MixedPrecisionBenchmark
    >>> from tests.fixtures import create_synthetic_model
    >>> model = create_synthetic_model()
    >>> bench = MixedPrecisionBenchmark(model)
    >>> results = bench.compare_all()
    >>> print(results)
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    
    strategy: str
    throughput_tokens_per_sec: float
    latency_ms_per_token: float
    memory_peak_mb: float
    memory_allocated_mb: float
    iterations: int
    warmup: int
    batch_size: int
    seq_len: int
    accuracy_match: bool = True
    notes: str = ""
    
    @property
    def speedup(self) -> float:
        """Speedup relative to 1.0 baseline."""
        return getattr(self, "_speedup", 1.0)
    
    @speedup.setter
    def speedup(self, value: float) -> None:
        self._speedup = value
        
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["speedup"] = self.speedup
        return d
    
    def __str__(self) -> str:
        return (
            f"{self.strategy:20s} | "
            f"{self.throughput_tokens_per_sec:7.1f} | "
            f"{self.latency_ms_per_token:7.1f} | "
            f"{self.memory_allocated_mb:6.0f}MB | "
            f"{self.speedup:5.1f}x"
        )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    warmup: int = 3
    iterations: int = 10
    batch_size: int = 1
    seq_len: int = 1
    check_accuracy: bool = True
    accuracy_atol: float = 1e-2
    accuracy_rtol: float = 1e-2
    
    # Strategies to benchmark
    strategies: list[str] = field(default_factory=lambda: [
        "slow_path",
        "fast_uniform",
        "fast_mixed",
        "hybrid",
        "max_bits_padded",
    ])


class MixedPrecisionBenchmark:
    """Benchmark harness for comparing MoE dispatch strategies.
    
    This class benchmarks different approaches to handling mixed-precision
    quantization in MoE layers:
    
    1. slow_path: Sequential per-expert dispatch (baseline)
    2. fast_uniform: Batched dispatch with uniform bits (ideal target)
    3. fast_mixed: Batched dispatch with per-projection bits (Strategy A)
    4. hybrid: Batched for common tuples, sequential for rare (Strategy C)
    5. max_bits_padded: Pad to max bits, single dispatch (Strategy B)
    
    Usage:
        >>> model = create_synthetic_model()
        >>> bench = MixedPrecisionBenchmark(model)
        >>> results = bench.compare_all()
        >>> bench.print_results(results)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: BenchmarkConfig | None = None,
        device: str = "mps",
    ):
        self.model = model
        self.config = config or BenchmarkConfig()
        self.device = device
        
        # Reference output for accuracy checking
        self._reference_output: torch.Tensor | None = None
        
    def _create_input(self) -> torch.Tensor:
        """Create input tensor for benchmarking."""
        hidden_dim = self.model.config.hidden_dim
        x = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        return x
    
    def _sync_device(self) -> None:
        """Synchronize device for accurate timing."""
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()
            
    def _get_memory_stats(self) -> tuple[float, float]:
        """Get memory statistics (peak, allocated) in MB."""
        if self.device == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024 / 1024
            # MPS doesn't have peak memory tracking, use allocated
            peak = allocated
        elif self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            allocated = peak = 0
        return peak, allocated
    
    def _clear_memory(self) -> None:
        """Clear GPU memory caches."""
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    def _run_benchmark(
        self,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        strategy: str,
    ) -> BenchmarkResult:
        """Run a single benchmark with the given forward function.
        
        Args:
            forward_fn: Function that takes input tensor and returns output
            strategy: Name of the strategy being benchmarked
            
        Returns:
            BenchmarkResult with timing and memory statistics
        """
        self._clear_memory()
        x = self._create_input()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup):
                _ = forward_fn(x)
                self._sync_device()
                
        # Record memory baseline
        self._clear_memory()
        self._sync_device()
        
        # Timed iterations
        times: list[float] = []
        output: torch.Tensor | None = None
        
        with torch.no_grad():
            for _ in range(self.config.iterations):
                self._sync_device()
                start = time.perf_counter()
                
                output = forward_fn(x)
                
                self._sync_device()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                
        # Memory stats
        peak_mb, allocated_mb = self._get_memory_stats()
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        tokens_per_iter = self.config.batch_size * self.config.seq_len
        throughput = tokens_per_iter / avg_time if avg_time > 0 else 0
        latency = (avg_time * 1000 / tokens_per_iter) if tokens_per_iter > 0 else 0
        
        # Check accuracy against reference
        accuracy_match = True
        if self.config.check_accuracy and self._reference_output is not None and output is not None:
            accuracy_match = torch.allclose(
                output,
                self._reference_output,
                atol=self.config.accuracy_atol,
                rtol=self.config.accuracy_rtol,
            )
            
        return BenchmarkResult(
            strategy=strategy,
            throughput_tokens_per_sec=throughput,
            latency_ms_per_token=latency,
            memory_peak_mb=peak_mb,
            memory_allocated_mb=allocated_mb,
            iterations=self.config.iterations,
            warmup=self.config.warmup,
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
            accuracy_match=accuracy_match,
        )
        
    def benchmark_slow_path(self) -> BenchmarkResult:
        """Benchmark per-expert sequential dispatch (current fallback).
        
        This is the baseline - sequential dispatch where each expert is
        called individually. Slow but always correct.
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            # Use model's default forward (sequential dispatch)
            return self.model(x)
            
        result = self._run_benchmark(forward, "slow_path")
        
        # Store reference output
        if self._reference_output is None:
            with torch.no_grad():
                self._reference_output = self.model(self._create_input())
                
        return result
    
    def benchmark_fast_uniform(self) -> BenchmarkResult:
        """Benchmark batched dispatch with uniform bits.
        
        This represents the ideal performance target - what we'd get if
        all experts had the same bit width. Uses the fused MoE kernel.
        """
        # For synthetic model, just use default forward (which is sequential)
        # In real testing, this would use a uniform-bits model
        def forward(x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
            
        result = self._run_benchmark(forward, "fast_uniform")
        result.notes = "Placeholder - requires uniform-bits model"
        return result
    
    def benchmark_fast_mixed(self) -> BenchmarkResult:
        """Benchmark batched dispatch with per-projection bits (Strategy A).
        
        Groups experts by (gate_bits, up_bits, down_bits) tuple and
        dispatches each group with the fused kernel using per-projection bits.
        """
        # Currently falls back to slow path - this will use real fast path
        # after kernel modifications are complete
        def forward(x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
            
        result = self._run_benchmark(forward, "fast_mixed")
        result.notes = "Requires per-projection bits kernel (Phase 1)"
        return result
    
    def benchmark_hybrid(self) -> BenchmarkResult:
        """Benchmark hybrid dispatch (Strategy C).
        
        Uses batched dispatch for common bit tuples and sequential
        dispatch for rare ones. Optimizes for the case where most
        selected experts share bit tuples.
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
            
        result = self._run_benchmark(forward, "hybrid")
        result.notes = "Requires hybrid dispatch implementation (Phase 3)"
        return result
    
    def benchmark_max_bits_padded(self) -> BenchmarkResult:
        """Benchmark max-bits padding (Strategy B).
        
        Pads all projections to max(gate, up, down) bits and uses
        single batched dispatch. Simpler but has compute overhead.
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            # 1. Dense layer (standard)
            # x is [batch, seq, hidden]
            h = x.half()
            h = h + self.model.dense_layer(h)
            
            # 2. MoE layer (Optimized Strategy B dispatch)
            # We bypass model.moe_layer.forward and implement batched logic here
            moe = self.model.moe_layer
            
            batch_shape = h.shape[:-1]
            h_flat = h.view(-1, moe.hidden_dim)
            
            # Route
            router_logits = moe.router(h_flat.float())
            routing_weights, selected_experts = torch.topk(
                F.softmax(router_logits, dim=-1),
                k=moe.top_k,
                dim=-1,
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            
            # Batched Dispatch Simulation
            out_flat = torch.zeros_like(h_flat)
            
            for expert_idx in range(moe.num_experts):
                # Find tokens routed to this expert
                idx_in_batch, k_idx = torch.where(selected_experts == expert_idx)
                
                if idx_in_batch.numel() > 0:
                    tokens = h_flat[idx_in_batch]
                    
                    # Run expert
                    expert = moe.experts[expert_idx]
                    expert_out = expert(tokens)
                    
                    # Weight and accumulate
                    weights = routing_weights[idx_in_batch, k_idx].unsqueeze(-1).half()
                    out_flat.index_add_(0, idx_in_batch, expert_out * weights)
            
            h = h + out_flat.view(*batch_shape, moe.hidden_dim)
            
            # 3. LM Head
            logits = self.model.lm_head(h.float())
            return logits.half()
            
        result = self._run_benchmark(forward, "max_bits_padded")
        result.notes = "Simulated batched dispatch (Strategy B)"
        return result
    
    def compare_all(self) -> list[BenchmarkResult]:
        """Run all configured benchmarks and return results.
        
        Returns:
            List of BenchmarkResult objects, sorted by throughput (descending)
        """
        results: list[BenchmarkResult] = []
        
        strategy_methods = {
            "slow_path": self.benchmark_slow_path,
            "fast_uniform": self.benchmark_fast_uniform,
            "fast_mixed": self.benchmark_fast_mixed,
            "hybrid": self.benchmark_hybrid,
            "max_bits_padded": self.benchmark_max_bits_padded,
        }
        
        # Run slow_path first to establish reference
        if "slow_path" in self.config.strategies:
            result = self.benchmark_slow_path()
            results.append(result)
            baseline_throughput = result.throughput_tokens_per_sec
        else:
            baseline_throughput = 1.0
            
        # Run remaining strategies
        for strategy in self.config.strategies:
            if strategy == "slow_path":
                continue
            if strategy in strategy_methods:
                result = strategy_methods[strategy]()
                result.speedup = (
                    result.throughput_tokens_per_sec / baseline_throughput
                    if baseline_throughput > 0 else 0
                )
                results.append(result)
                
        # Set baseline speedup
        for r in results:
            if r.strategy == "slow_path":
                r.speedup = 1.0
                
        return sorted(results, key=lambda r: r.throughput_tokens_per_sec, reverse=True)
    
    def print_results(self, results: list[BenchmarkResult]) -> None:
        """Print results in a formatted table."""
        print("\nMixed Precision Benchmark Results")
        print("=" * 70)
        print(f"{'Strategy':20s} | {'tok/s':>7s} | {'ms/tok':>7s} | {'Memory':>8s} | {'Speedup':>7s}")
        print("-" * 70)
        
        for result in results:
            print(result)
            
        print("=" * 70)
        
        # Print notes
        notes_printed = False
        for result in results:
            if result.notes:
                if not notes_printed:
                    print("\nNotes:")
                    notes_printed = True
                print(f"  {result.strategy}: {result.notes}")
                
        # Print accuracy warnings
        inaccurate = [r for r in results if not r.accuracy_match]
        if inaccurate:
            print("\n⚠️  Accuracy mismatch detected:")
            for r in inaccurate:
                print(f"  - {r.strategy}")
                
    def save_results(
        self,
        results: list[BenchmarkResult],
        path: str | Path,
    ) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": asdict(self.config),
            "results": [r.to_dict() for r in results],
            "model_config": {
                "hidden_dim": self.model.config.hidden_dim,
                "intermediate_dim": self.model.config.intermediate_dim,
                "num_experts": self.model.config.num_experts,
                "top_k": self.model.config.top_k,
            },
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"\nResults saved to: {path}")


def run_quick_benchmark() -> None:
    """Run a quick benchmark with default settings."""
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from tests.fixtures.synthetic_mixed_moe import create_synthetic_model
    
    print("Creating synthetic model...")
    model = create_synthetic_model(device="mps")
    
    print(f"Model config: {model.config.hidden_dim}d, {model.config.num_experts} experts")
    print(f"Bit distribution: {model.get_bit_distribution()['bit_tuple_counts']}")
    
    print("\nRunning benchmark...")
    config = BenchmarkConfig(
        warmup=2,
        iterations=5,
        strategies=["slow_path", "fast_mixed", "hybrid", "fast_uniform"],  # Quick comparison
    )
    
    bench = MixedPrecisionBenchmark(model, config)
    results = bench.compare_all()
    bench.print_results(results)
    
    return results


if __name__ == "__main__":
    run_quick_benchmark()
