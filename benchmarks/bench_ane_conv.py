"""
Benchmark ANE vs MPS for 1D convolutions.

Tests performance for different convolution scenarios:
- Pointwise conv (kernel=1)
- Depthwise conv (kernel=31, groups=hidden)
- Full Conformer conv module

Measures:
- Latency
- Memory bandwidth
- Transfer overhead (CPU<->ANE<->MPS)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# Check if ANE is available
try:
    import torch_neuron

    HAS_ANE = True
except ImportError:
    HAS_ANE = False


@dataclass
class ConvBenchmarkResult:
    """Result from convolution benchmark."""

    name: str
    seq_len: int
    hidden: int
    kernel: int
    groups: int
    mps_time_ms: float
    ane_time_ms: float | None = None
    speedup: float | None = None
    memory_gb_mps: float = 0.0
    memory_gb_ane: float = 0.0
    transfer_overhead_ms: float = 0.0


class ConformerConv1d(nn.Module):
    """Conformer-style convolution module with GLU activation."""

    def __init__(self, hidden: int, kernel: int = 31, expansion: int = 2):
        super().__init__()
        self.hidden = hidden
        self.kernel = kernel
        self.expansion = expansion

        # Pointwise conv for channel expansion
        self.pointwise_conv1 = nn.Conv1d(hidden, hidden * expansion, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            hidden * expansion // 2,
            hidden * expansion // 2,
            kernel_size=kernel,
            groups=hidden * expansion // 2,
            padding=kernel // 2,
        )

        # Batch norm and activation
        self.batch_norm = nn.BatchNorm1d(hidden * expansion // 2)
        self.activation = nn.SiLU()

        # Pointwise conv to project back
        self.pointwise_conv2 = nn.Conv1d(hidden * expansion // 2, hidden, kernel_size=1)

        # Layer norm for residual connection
        self.layer_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden]

        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        # Store residual
        residual = x

        # Layer norm
        x = self.layer_norm(x)

        # Transpose for conv1d: [batch, hidden, seq_len]
        x = x.transpose(1, 2)

        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise conv back to original dim
        x = self.pointwise_conv2(x)

        # Transpose back: [batch, seq_len, hidden]
        x = x.transpose(1, 2)

        # Add residual
        return x + residual


def time_mps_conv(
    fn: Callable[[], torch.Tensor],
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Time a convolution function on MPS."""

    # Warmup
    for _ in range(warmup):
        _ = fn()
        torch.mps.synchronize()

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = fn()
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

    return sum(times) / len(times)


def time_ane_conv(
    fn: Callable[[], torch.Tensor],
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Time a convolution function on ANE."""

    if not HAS_ANE:
        raise RuntimeError("ANE not available. Install torch-neuron.")

    # Transfer to CPU first (simulating typical workflow)
    def cpu_to_ane_wrapper():
        result = fn()
        # Force transfer to CPU then back to ANE
        cpu_result = result.cpu()
        return cpu_result.to("ane")

    # Warmup
    for _ in range(warmup):
        _ = cpu_to_ane_wrapper()

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = cpu_to_ane_wrapper()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

    return sum(times) / len(times)


def measure_memory_usage(
    fn: Callable[[], torch.Tensor],
    device: str,
) -> float:
    """Measure memory usage in GB for a function."""

    if device == "mps":
        torch.mps.empty_cache()
        initial_memory = torch.mps.current_allocated_memory()

        _ = fn()
        torch.mps.synchronize()

        peak_memory = torch.mps.current_allocated_memory()
        return (peak_memory - initial_memory) / (1024**3)

    elif device == "ane" and HAS_ANE:
        # ANE memory measurement is more approximate
        # Use a proxy measurement based on tensor sizes
        return 0.0  # Placeholder

    return 0.0


def benchmark_pointwise_conv(
    seq_len: int,
    hidden: int,
    warmup: int = 10,
    iterations: int = 100,
) -> ConvBenchmarkResult:
    """Benchmark pointwise convolution."""

    # Create input
    x_mps = torch.randn(1, seq_len, hidden, device="mps", dtype=torch.float32)

    # Create pointwise conv layer
    conv = nn.Conv1d(hidden, hidden, kernel_size=1).to("mps")

    def mps_fn():
        x_conv = x_mps.transpose(1, 2)  # [batch, hidden, seq_len]
        return conv(x_conv).transpose(1, 2)

    # Benchmark MPS
    mps_time = time_mps_conv(mps_fn, warmup, iterations)
    mps_memory = measure_memory_usage(mps_fn, "mps")

    # Benchmark ANE if available
    ane_time = None
    ane_memory = 0.0
    if HAS_ANE:
        x_ane = x_mps.cpu().to("ane")
        conv_ane = nn.Conv1d(hidden, hidden, kernel_size=1).to("ane")

        def ane_fn():
            x_conv = x_ane.transpose(1, 2)
            return conv_ane(x_conv).transpose(1, 2)

        ane_time = time_ane_conv(ane_fn, warmup, iterations)
        ane_memory = measure_memory_usage(ane_fn, "ane")

    # Calculate transfer overhead
    transfer_time = 0.0
    if HAS_ANE:
        start = time.perf_counter()
        for _ in range(10):
            x_transfer = x_mps.cpu().to("ane")
            _ = x_transfer.cpu()
        transfer_time = (time.perf_counter() - start) * 1000.0 / 10

    speedup = mps_time / ane_time if ane_time is not None else None

    return ConvBenchmarkResult(
        name="pointwise_conv",
        seq_len=seq_len,
        hidden=hidden,
        kernel=1,
        groups=1,
        mps_time_ms=mps_time,
        ane_time_ms=ane_time,
        speedup=speedup,
        memory_gb_mps=mps_memory,
        memory_gb_ane=ane_memory,
        transfer_overhead_ms=transfer_time,
    )


def benchmark_depthwise_conv(
    seq_len: int,
    hidden: int,
    kernel: int = 31,
    warmup: int = 10,
    iterations: int = 100,
) -> ConvBenchmarkResult:
    """Benchmark depthwise convolution."""

    # Create input
    x_mps = torch.randn(1, seq_len, hidden, device="mps", dtype=torch.float32)

    # Create depthwise conv layer
    conv = nn.Conv1d(hidden, hidden, kernel_size=kernel, groups=hidden, padding=kernel // 2).to(
        "mps"
    )

    def mps_fn():
        x_conv = x_mps.transpose(1, 2)  # [batch, hidden, seq_len]
        return conv(x_conv).transpose(1, 2)

    # Benchmark MPS
    mps_time = time_mps_conv(mps_fn, warmup, iterations)
    mps_memory = measure_memory_usage(mps_fn, "mps")

    # Benchmark ANE if available
    ane_time = None
    ane_memory = 0.0
    if HAS_ANE:
        x_ane = x_mps.cpu().to("ane")
        conv_ane = nn.Conv1d(
            hidden, hidden, kernel_size=kernel, groups=hidden, padding=kernel // 2
        ).to("ane")

        def ane_fn():
            x_conv = x_ane.transpose(1, 2)
            return conv_ane(x_conv).transpose(1, 2)

        ane_time = time_ane_conv(ane_fn, warmup, iterations)
        ane_memory = measure_memory_usage(ane_fn, "ane")

    # Calculate transfer overhead
    transfer_time = 0.0
    if HAS_ANE:
        start = time.perf_counter()
        for _ in range(10):
            x_transfer = x_mps.cpu().to("ane")
            _ = x_transfer.cpu()
        transfer_time = (time.perf_counter() - start) * 1000.0 / 10

    speedup = mps_time / ane_time if ane_time is not None else None

    return ConvBenchmarkResult(
        name="depthwise_conv",
        seq_len=seq_len,
        hidden=hidden,
        kernel=kernel,
        groups=hidden,
        mps_time_ms=mps_time,
        ane_time_ms=ane_time,
        speedup=speedup,
        memory_gb_mps=mps_memory,
        memory_gb_ane=ane_memory,
        transfer_overhead_ms=transfer_time,
    )


def benchmark_conformer_conv(
    seq_len: int,
    hidden: int,
    kernel: int = 31,
    warmup: int = 10,
    iterations: int = 100,
) -> ConvBenchmarkResult:
    """Benchmark full Conformer convolution module."""

    # Create input
    x_mps = torch.randn(1, seq_len, hidden, device="mps", dtype=torch.float32)

    # Create Conformer conv module
    conformer_conv = ConformerConv1d(hidden, kernel).to("mps")

    def mps_fn():
        return conformer_conv(x_mps)

    # Benchmark MPS
    mps_time = time_mps_conv(mps_fn, warmup, iterations)
    mps_memory = measure_memory_usage(mps_fn, "mps")

    # Benchmark ANE if available
    ane_time = None
    ane_memory = 0.0
    if HAS_ANE:
        x_ane = x_mps.cpu().to("ane")
        conformer_conv_ane = ConformerConv1d(hidden, kernel).to("ane")

        def ane_fn():
            return conformer_conv_ane(x_ane)

        ane_time = time_ane_conv(ane_fn, warmup, iterations)
        ane_memory = measure_memory_usage(ane_fn, "ane")

    # Calculate transfer overhead
    transfer_time = 0.0
    if HAS_ANE:
        start = time.perf_counter()
        for _ in range(10):
            x_transfer = x_mps.cpu().to("ane")
            _ = x_transfer.cpu()
        transfer_time = (time.perf_counter() - start) * 1000.0 / 10

    speedup = mps_time / ane_time if ane_time is not None else None

    return ConvBenchmarkResult(
        name="conformer_conv",
        seq_len=seq_len,
        hidden=hidden,
        kernel=kernel,
        groups=hidden // 2,  # Approximate groups in depthwise portion
        mps_time_ms=mps_time,
        ane_time_ms=ane_time,
        speedup=speedup,
        memory_gb_mps=mps_memory,
        memory_gb_ane=ane_memory,
        transfer_overhead_ms=transfer_time,
    )


def benchmark_ane_vs_mps_conv():
    """Main benchmark function comparing ANE vs MPS for conv1d operations."""

    print("ANE vs MPS Conv1D Benchmark")
    print("=" * 50)
    print(f"ANE Available: {HAS_ANE}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print()

    # Conformer conv params
    hidden = 512
    kernel = 31
    seq_lens = [100, 500, 1000, 2000]  # frames

    results = []

    print(f"Testing with hidden={hidden}, kernel={kernel}")
    print("-" * 50)

    for seq_len in seq_lens:
        print(f"\nSequence Length: {seq_len}")

        # Pointwise conv
        result_pointwise = benchmark_pointwise_conv(seq_len, hidden)
        results.append(result_pointwise)
        print(f"  Pointwise:  MPS={result_pointwise.mps_time_ms:.2f}ms", end="")
        if result_pointwise.ane_time_ms:
            print(f", ANE={result_pointwise.ane_time_ms:.2f}ms", end="")
            print(f", Speedup={result_pointwise.speedup:.2f}x", end="")
        print()

        # Depthwise conv
        result_depthwise = benchmark_depthwise_conv(seq_len, hidden, kernel)
        results.append(result_depthwise)
        print(f"  Depthwise:  MPS={result_depthwise.mps_time_ms:.2f}ms", end="")
        if result_depthwise.ane_time_ms:
            print(f", ANE={result_depthwise.ane_time_ms:.2f}ms", end="")
            print(f", Speedup={result_depthwise.speedup:.2f}x", end="")
        print()

        # Full Conformer conv
        result_conformer = benchmark_conformer_conv(seq_len, hidden, kernel)
        results.append(result_conformer)
        print(f"  Conformer:  MPS={result_conformer.mps_time_ms:.2f}ms", end="")
        if result_conformer.ane_time_ms:
            print(f", ANE={result_conformer.ane_time_ms:.2f}ms", end="")
            print(f", Speedup={result_conformer.speedup:.2f}x", end="")
        print()

        if HAS_ANE:
            transfer = result_pointwise.transfer_overhead_ms
            print(f"  Transfer:    {transfer:.2f}ms (CPU<->ANE)")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Type':<12} {'Seq':<6} {'Hidden':<7} {'Kernel':<7} {'Groups':<7} "
        f"{'MPS(ms)':<9} {'ANE(ms)':<9} {'Speedup':<9} {'Trans(ms)':<9}"
    )
    print("-" * 80)

    for r in results:
        ane_str = f"{r.ane_time_ms:.2f}" if r.ane_time_ms else "N/A"
        speedup_str = f"{r.speedup:.2f}x" if r.speedup else "N/A"

        print(
            f"{r.name:<12} {r.seq_len:<6} {r.hidden:<7} {r.kernel:<7} {r.groups:<7} "
            f"{r.mps_time_ms:<9.2f} {ane_str:<9} {speedup_str:<9} "
            f"{r.transfer_overhead_ms:<9.2f}"
        )

    # Memory analysis
    print("\n" + "=" * 50)
    print("MEMORY USAGE (GB)")
    print("-" * 50)

    for r in results:
        if r.memory_gb_mps > 0 or r.memory_gb_ane > 0:
            print(
                f"{r.name:<12} seq={r.seq_len}: "
                f"MPS={r.memory_gb_mps:.3f}GB, ANE={r.memory_gb_ane:.3f}GB"
            )

    return results


if __name__ == "__main__":
    benchmark_ane_vs_mps_conv()
