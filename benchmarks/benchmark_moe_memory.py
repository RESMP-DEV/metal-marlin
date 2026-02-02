#!/usr/bin/env python3
"""Memory usage benchmarks for MoE inference.

Measures peak memory for:
- Model loading (64 experts x 3 projections)
- Single forward pass at various batch sizes
- Cached weight buffers vs fresh allocation

Compares against theoretical minimum:
- 3-bit weights: 3/8 bytes per weight
- FP16 activations: 2 bytes per element
- Calculate memory efficiency ratio

Profile using torch.mps.driver_allocated_memory()
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Type alias for model to avoid import cycle
TrellisModel = Any


@dataclass
class MemorySnapshot:
    """Memory snapshot at a specific point."""

    label: str
    allocated_mb: float
    driver_mb: float


@dataclass
class MoEMemoryReport:
    """Complete memory report for MoE model."""

    # Model configuration
    num_experts: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_moe_layers: int
    bits: int

    # Weight memory breakdown
    packed_weights_mb: float
    scales_mb: float
    su_sv_mb: float
    grid_mb: float
    total_expert_weights_mb: float
    shared_expert_weights_mb: float
    router_weights_mb: float

    # Theoretical values
    theoretical_min_mb: float
    memory_efficiency: float

    # Runtime memory
    model_load_mb: float
    cached_buffers_mb: float
    activation_peak_mb: dict[int, float]  # batch_size -> peak MB

    def print_report(self) -> None:
        """Print formatted memory report."""
        print("=" * 70)
        print("MoE Memory Benchmark Report")
        print("=" * 70)

        print("\nModel Configuration:")
        print(f"  Experts:           {self.num_experts}")
        print(f"  Hidden size:       {self.hidden_size}")
        print(f"  Intermediate:      {self.intermediate_size}")
        print(f"  Total layers:      {self.num_layers}")
        print(f"  MoE layers:        {self.num_moe_layers}")
        print(f"  Bits per weight:   {self.bits}")

        print("\nWeight Memory Breakdown (per MoE layer):")
        print(f"  Packed weights:    {self.packed_weights_mb:.2f} MB")
        print(f"  Scales:            {self.scales_mb:.2f} MB")
        print(f"  SU/SV vectors:     {self.su_sv_mb:.2f} MB")
        grid_kb = self.grid_mb * 1024
        print(f"  Codebook grid:     {grid_kb:.2f} KB ({2**self.bits} levels)")
        print("  ---")
        print(f"  Expert weights:    {self.total_expert_weights_mb:.2f} MB")
        print(f"  Shared expert:     {self.shared_expert_weights_mb:.2f} MB")
        print(f"  Router weights:    {self.router_weights_mb:.2f} MB")

        print("\nTheoretical vs Actual:")
        print(f"  Theoretical min:   {self.theoretical_min_mb:.2f} MB")
        total_per_layer = (
            self.total_expert_weights_mb
            + self.shared_expert_weights_mb
            + self.router_weights_mb
        )
        print(f"  Actual per layer:  {total_per_layer:.2f} MB")
        print(f"  Efficiency ratio:  {self.memory_efficiency:.2%}")

        print("\nRuntime Memory:")
        print(f"  Model load total:  {self.model_load_mb:.2f} MB")
        if self.cached_buffers_mb > 0:
            print(f"  Cached buffers:    {self.cached_buffers_mb:.2f} MB")

        print("\nActivation Peak Memory (forward pass):")
        for batch_size, peak_mb in sorted(self.activation_peak_mb.items()):
            print(f"  Batch {batch_size:4d}:         {peak_mb:.2f} MB")

        print("=" * 70)


def get_mps_memory() -> tuple[float, float]:
    """Get MPS memory in MB (allocated, driver)."""
    allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
    driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
    return allocated, driver


def sync_and_clear() -> None:
    """Synchronize MPS and clear caches."""
    torch.mps.synchronize()
    torch.mps.empty_cache()
    gc.collect()


def compute_theoretical_weights_mb(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bits: int,
) -> float:
    """Compute theoretical minimum weight memory.

    For each expert:
    - gate_proj: [hidden, intermediate] = hidden * intermediate weights
    - up_proj: [hidden, intermediate] = hidden * intermediate weights
    - down_proj: [intermediate, hidden] = intermediate * hidden weights
    Total per expert: 2 * hidden * intermediate + intermediate * hidden
                    = 3 * hidden * intermediate

    At bits-per-weight: 3 * hidden * intermediate * bits / 8 bytes
    """
    weights_per_expert = 3 * hidden_size * intermediate_size
    total_weights = num_experts * weights_per_expert

    # Add shared expert (same size as one regular expert)
    total_weights += weights_per_expert

    # Add router: [hidden, num_experts] in FP16
    router_bytes = hidden_size * num_experts * 2  # FP16

    # Weight bytes at target bit width
    weight_bytes = total_weights * bits / 8

    total_bytes = weight_bytes + router_bytes
    return total_bytes / (1024 * 1024)


def compute_buffer_sizes(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bits: int,
    tile_dim: int = 16,
    group_size: int = 128,
) -> dict[str, float]:
    """Compute actual buffer sizes for TrellisLinear weights.

    Returns dictionary with sizes in MB for each component.
    """
    # Packed indices: [tiles_k, tiles_n, packed_bytes] uint8
    # For gate/up: K=hidden, N=intermediate
    # For down: K=intermediate, N=hidden
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]

    tiles_gate_k = (hidden_size + tile_dim - 1) // tile_dim
    tiles_gate_n = (intermediate_size + tile_dim - 1) // tile_dim
    tiles_down_k = (intermediate_size + tile_dim - 1) // tile_dim
    tiles_down_n = (hidden_size + tile_dim - 1) // tile_dim

    gate_packed = tiles_gate_k * tiles_gate_n * packed_bytes
    up_packed = gate_packed  # Same shape
    down_packed = tiles_down_k * tiles_down_n * packed_bytes

    total_packed = num_experts * (gate_packed + up_packed + down_packed)

    # Scales: [n_groups, out_features] float32
    # Groups are along input dimension
    n_groups_gate = (hidden_size + group_size - 1) // group_size
    n_groups_down = (intermediate_size + group_size - 1) // group_size

    gate_scales = n_groups_gate * intermediate_size * 4  # float32
    up_scales = gate_scales
    down_scales = n_groups_down * hidden_size * 4

    total_scales = num_experts * (gate_scales + up_scales + down_scales)

    # SU/SV vectors: float32
    # gate/up: su=[hidden], sv=[intermediate]
    # down: su=[intermediate], sv=[hidden]
    gate_su_sv = (hidden_size + intermediate_size) * 4
    up_su_sv = gate_su_sv
    down_su_sv = (intermediate_size + hidden_size) * 4

    total_su_sv = num_experts * (gate_su_sv + up_su_sv + down_su_sv)

    # Codebook grid: [n_levels] float32
    n_levels = 2**bits
    grid_size = n_levels * 4

    return {
        "packed_weights_bytes": total_packed,
        "scales_bytes": total_scales,
        "su_sv_bytes": total_su_sv,
        "grid_bytes": grid_size,
        "packed_weights_mb": total_packed / (1024 * 1024),
        "scales_mb": total_scales / (1024 * 1024),
        "su_sv_mb": total_su_sv / (1024 * 1024),
        "grid_mb": grid_size / (1024 * 1024),
    }


def measure_model_loading(model_path: str | Path) -> tuple[float, dict[str, Any], TrellisModel]:
    """Measure memory during model loading.

    Returns (peak_mb, config_dict, model).
    """
    from metal_marlin.trellis.model import TrellisForCausalLM

    sync_and_clear()
    _, baseline = get_mps_memory()

    model = TrellisForCausalLM.from_pretrained(str(model_path), device="mps")

    sync_and_clear()
    _, after_load = get_mps_memory()

    config = model.config

    # Extract config values
    config_dict: dict[str, Any] = {
        "num_experts": config.num_experts,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.moe_intermediate_size or config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "first_moe_layer": config.first_moe_layer,
        "bits": config.quantization_bits,
    }

    # Count MoE layers
    num_moe_layers = sum(
        1 for i in range(config.num_hidden_layers) if config.is_moe_layer(i)
    )
    config_dict["num_moe_layers"] = num_moe_layers

    peak_mb = after_load - baseline

    return peak_mb, config_dict, model


def measure_cached_buffers(model: TrellisModel) -> float:
    """Measure memory used by cached weight buffers.

    Returns delta in MB.
    """
    sync_and_clear()
    _, before = get_mps_memory()

    # Access one MoE layer to trigger cached buffer creation
    for layer in model.model.layers:
        if hasattr(layer.mlp, "_get_cached_buffers"):
            _ = layer.mlp._get_cached_buffers()
            break

    sync_and_clear()
    _, after = get_mps_memory()

    return after - before


def measure_forward_pass(
    model: TrellisModel,
    batch_sizes: list[int],
    seq_len: int = 128,
) -> dict[int, float]:
    """Measure peak activation memory for forward passes.

    Returns dict mapping batch_size to peak MB delta.
    """
    results = {}

    vocab_size = model.config.vocab_size

    for batch_size in batch_sizes:
        sync_and_clear()
        _, baseline = get_mps_memory()

        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="mps")

        # Forward pass
        with torch.inference_mode():
            _ = model(input_ids)

        sync_and_clear()
        _, after = get_mps_memory()

        results[batch_size] = after - baseline

        # Clean up
        del input_ids
        sync_and_clear()

    return results


def measure_fresh_vs_cached(
    model: TrellisModel, batch_size: int = 1, seq_len: int = 32
) -> dict[str, float]:
    """Compare memory between fresh allocation and cached buffers.

    Returns dict with fresh_mb, cached_mb, savings_mb.
    """
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="mps")

    # First run - may create cached buffers
    with torch.inference_mode():
        _ = model(input_ids)
    sync_and_clear()

    # Measure with cached buffers
    _, baseline = get_mps_memory()
    with torch.inference_mode():
        _ = model(input_ids)
    sync_and_clear()
    _, after_cached = get_mps_memory()
    cached_mb = after_cached - baseline

    return {
        "cached_mb": cached_mb,
    }


def run_synthetic_benchmark(
    num_experts: int = 64,
    hidden_size: int = 3584,
    intermediate_size: int = 18944,
    bits: int = 3,
) -> MoEMemoryReport:
    """Run benchmark with synthetic configuration (no model loading).

    Useful for testing without a real model checkpoint.
    """
    # Compute buffer sizes
    buf_sizes = compute_buffer_sizes(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=bits,
    )

    # Shared expert (same size as one expert)
    shared_sizes = compute_buffer_sizes(
        num_experts=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=bits,
    )

    # Router: [hidden, num_experts] FP16
    router_mb = (hidden_size * num_experts * 2) / (1024 * 1024)

    # Theoretical minimum
    theoretical_mb = compute_theoretical_weights_mb(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=bits,
    )

    # Total actual
    total_expert_mb = (
        buf_sizes["packed_weights_mb"]
        + buf_sizes["scales_mb"]
        + buf_sizes["su_sv_mb"]
        + buf_sizes["grid_mb"]
    )
    total_shared_mb = (
        shared_sizes["packed_weights_mb"]
        + shared_sizes["scales_mb"]
        + shared_sizes["su_sv_mb"]
        + shared_sizes["grid_mb"]
    )

    actual_total = total_expert_mb + total_shared_mb + router_mb

    return MoEMemoryReport(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=1,  # Synthetic single layer
        num_moe_layers=1,
        bits=bits,
        packed_weights_mb=buf_sizes["packed_weights_mb"],
        scales_mb=buf_sizes["scales_mb"],
        su_sv_mb=buf_sizes["su_sv_mb"],
        grid_mb=buf_sizes["grid_mb"],
        total_expert_weights_mb=total_expert_mb,
        shared_expert_weights_mb=total_shared_mb,
        router_weights_mb=router_mb,
        theoretical_min_mb=theoretical_mb,
        memory_efficiency=theoretical_mb / actual_total if actual_total > 0 else 0,
        model_load_mb=0,
        cached_buffers_mb=0,
        activation_peak_mb={},
    )


def run_full_benchmark(
    model_path: str | Path,
    batch_sizes: list[int] | None = None,
) -> MoEMemoryReport:
    """Run full benchmark with real model.

    Args:
        model_path: Path to trellis-quantized model directory.
        batch_sizes: List of batch sizes to test for activation memory.
            Default: [1, 4, 16, 64]

    Returns:
        MoEMemoryReport with all measurements.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64]

    print("Loading model...")
    load_mb, config, model = measure_model_loading(model_path)

    print("Measuring cached buffer allocation...")
    cached_mb = measure_cached_buffers(model)

    print("Measuring forward pass memory...")
    activation_peaks = measure_forward_pass(model, batch_sizes)

    # Compute buffer sizes based on config
    buf_sizes = compute_buffer_sizes(
        num_experts=config["num_experts"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        bits=config["bits"],
    )

    shared_sizes = compute_buffer_sizes(
        num_experts=1,
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        bits=config["bits"],
    )

    router_mb = (config["hidden_size"] * config["num_experts"] * 2) / (1024 * 1024)

    theoretical_mb = compute_theoretical_weights_mb(
        num_experts=config["num_experts"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        bits=config["bits"],
    )

    total_expert_mb = (
        buf_sizes["packed_weights_mb"]
        + buf_sizes["scales_mb"]
        + buf_sizes["su_sv_mb"]
        + buf_sizes["grid_mb"]
    )
    total_shared_mb = (
        shared_sizes["packed_weights_mb"]
        + shared_sizes["scales_mb"]
        + shared_sizes["su_sv_mb"]
        + shared_sizes["grid_mb"]
    )

    actual_per_layer = total_expert_mb + total_shared_mb + router_mb

    return MoEMemoryReport(
        num_experts=config["num_experts"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_layers=config["num_layers"],
        num_moe_layers=config["num_moe_layers"],
        bits=config["bits"],
        packed_weights_mb=buf_sizes["packed_weights_mb"],
        scales_mb=buf_sizes["scales_mb"],
        su_sv_mb=buf_sizes["su_sv_mb"],
        grid_mb=buf_sizes["grid_mb"],
        total_expert_weights_mb=total_expert_mb,
        shared_expert_weights_mb=total_shared_mb,
        router_weights_mb=router_mb,
        theoretical_min_mb=theoretical_mb,
        memory_efficiency=theoretical_mb / actual_per_layer if actual_per_layer > 0 else 0,
        model_load_mb=load_mb,
        cached_buffers_mb=cached_mb,
        activation_peak_mb=activation_peaks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory usage benchmarks for MoE inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trellis-quantized model directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic benchmark without model loading",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=64,
        help="Number of experts for synthetic benchmark",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=3584,
        help="Hidden size for synthetic benchmark",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=18944,
        help="Intermediate size for synthetic benchmark",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Quantization bits",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Batch sizes to test for forward pass",
    )

    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend required for this benchmark")

    if args.synthetic:
        report = run_synthetic_benchmark(
            num_experts=args.num_experts,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            bits=args.bits,
        )
        report.print_report()
    elif args.model:
        report = run_full_benchmark(
            model_path=args.model,
            batch_sizes=args.batch_sizes,
        )
        report.print_report()
    else:
        # Default: synthetic GLM-4.7-Flash config
        print("Running synthetic benchmark with GLM-4.7-Flash config...")
        print("(Use --model PATH for full benchmark with real model)")
        print()
        report = run_synthetic_benchmark(
            num_experts=64,
            hidden_size=3584,
            intermediate_size=18944,
            bits=3,
        )
        report.print_report()


if __name__ == "__main__":
    main()
