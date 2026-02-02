#!/usr/bin/env python3
"""Debug MoE kernel scaling issues.

Tests progressively larger configurations to find where the kernel breaks.
Isolates: num_experts, hidden_dim, intermediate_dim, batch_size, top_k.
"""

import argparse
import time
from dataclasses import dataclass

import torch


@dataclass
class MoEConfig:
    """MoE configuration for testing."""

    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    top_k: int
    batch_size: int
    bits: int = 3

    def __str__(self) -> str:
        return (
            f"experts={self.num_experts}, hidden={self.hidden_dim}, "
            f"inter={self.intermediate_dim}, top_k={self.top_k}, batch={self.batch_size}"
        )


def test_moe_config(config: MoEConfig, timeout_sec: float = 5.0, use_fast: bool = True) -> dict:
    """Test a single MoE configuration.

    Returns dict with: success, nan, time_ms, error
    """
    from metal_marlin.trellis.testing import create_mock_moe_mlp

    result = {
        "config": str(config),
        "success": False,
        "nan": None,
        "time_ms": None,
        "error": None,
    }

    try:
        # Create MoE layer
        moe = create_mock_moe_mlp(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            num_experts=config.num_experts,
            num_experts_per_tok=config.top_k,
            bits=config.bits,
            device="mps",
        )
        moe._use_fast_moe = use_fast

        # Create input
        x = torch.randn(config.batch_size, config.hidden_dim, dtype=torch.float16, device="mps")

        # Run with timeout (MPS doesn't have native timeout, so we use signal)
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Kernel exceeded {timeout_sec}s")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)

        try:
            torch.mps.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                output = moe(x)

            torch.mps.synchronize()
            end = time.perf_counter()

            signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel alarm

            result["success"] = True
            result["nan"] = output.isnan().any().item()
            result["time_ms"] = (end - start) * 1000

        except TimeoutError as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result["error"] = str(e)

    except Exception as e:
        result["error"] = str(e)

    return result


def run_scaling_tests(max_experts: int = 64, verbose: bool = True) -> list[dict]:
    """Run scaling tests to find where kernel breaks."""
    results = []

    # Test configurations in increasing order of complexity
    configs = [
        # Baseline - should always work
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=4, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=4, top_k=2, batch_size=4),
        # Scale hidden_dim
        MoEConfig(hidden_dim=512, intermediate_dim=1024, num_experts=4, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=1024, intermediate_dim=2048, num_experts=4, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=4, top_k=2, batch_size=1),
        # Scale num_experts
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=8, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=16, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=32, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=64, top_k=2, batch_size=1),
        # Scale top_k
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=16, top_k=4, batch_size=1),
        MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=16, top_k=8, batch_size=1),
        # Real model dimensions (GLM-4.7-Flash)
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=8, top_k=2, batch_size=1),
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=16, top_k=4, batch_size=1),
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=32, top_k=8, batch_size=1),
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=64, top_k=8, batch_size=1),
        # Batch scaling with real dims
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=64, top_k=8, batch_size=4),
        MoEConfig(hidden_dim=2048, intermediate_dim=1536, num_experts=64, top_k=8, batch_size=10),
    ]

    # Filter by max_experts
    configs = [c for c in configs if c.num_experts <= max_experts]

    print("=" * 80)
    print("MoE Kernel Scaling Tests")
    print("=" * 80)

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Testing: {config}")

        # Test fast path
        result = test_moe_config(config, timeout_sec=10.0, use_fast=True)
        results.append(result)

        status = "✓" if result["success"] and not result["nan"] else "✗"
        if result["success"]:
            nan_str = "NaN!" if result["nan"] else "OK"
            print(f"  Fast: {status} {nan_str} ({result['time_ms']:.2f}ms)")
        else:
            print(f"  Fast: {status} ERROR: {result['error']}")

        # If fast failed, also test slow for comparison
        if not result["success"] or result["nan"]:
            slow_result = test_moe_config(config, timeout_sec=30.0, use_fast=False)
            if slow_result["success"]:
                nan_str = "NaN!" if slow_result["nan"] else "OK"
                print(f"  Slow: ✓ {nan_str} ({slow_result['time_ms']:.2f}ms)")
            else:
                print(f"  Slow: ✗ ERROR: {slow_result['error']}")

    return results


def test_single_expert(config: MoEConfig) -> None:
    """Test a single expert in isolation (no routing)."""
    import torch.nn.functional as F

    from metal_marlin.trellis.testing import create_mock_moe_mlp

    print(f"\nTesting single expert with config: {config}")

    moe = create_mock_moe_mlp(
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_experts=config.num_experts,
        num_experts_per_tok=config.top_k,
        bits=config.bits,
        device="mps",
    )

    x = torch.randn(1, config.hidden_dim, dtype=torch.float16, device="mps")

    for expert_id in range(min(4, config.num_experts)):
        expert = moe.experts[expert_id]

        with torch.inference_mode():
            gate = expert.gate_proj(x)
            up = expert.up_proj(x)
            swiglu = F.silu(gate) * up
            down = expert.down_proj(swiglu)

        print(f"  Expert {expert_id}:")
        print(
            f"    Gate: NaN={gate.isnan().any().item()}, range=[{gate.min():.4f}, {gate.max():.4f}]"
        )
        print(f"    Up: NaN={up.isnan().any().item()}, range=[{up.min():.4f}, {up.max():.4f}]")
        print(
            f"    SwiGLU: NaN={swiglu.isnan().any().item()}, range=[{swiglu.min():.4f}, {swiglu.max():.4f}]"
        )
        print(
            f"    Down: NaN={down.isnan().any().item()}, range=[{down.min():.4f}, {down.max():.4f}]"
        )


def test_dispatch_parameters(config: MoEConfig) -> None:
    """Print dispatch parameters for debugging."""
    print(f"\nDispatch parameters for: {config}")

    MOE_TILE_N = 64
    MOE_TILE_K = 16
    TRELLIS_TILE = 16

    hidden_dim = config.hidden_dim
    intermediate_dim = config.intermediate_dim
    num_experts = config.num_experts

    # Grid dimensions
    grid_x = (hidden_dim + MOE_TILE_N - 1) // MOE_TILE_N
    grid_y = config.batch_size
    grid_z = config.top_k
    total_threadgroups = grid_x * grid_y * grid_z

    # Work per threadgroup
    num_intermediate_chunks = (intermediate_dim + MOE_TILE_N - 1) // MOE_TILE_N
    num_k_tiles_hidden = (hidden_dim + MOE_TILE_K - 1) // MOE_TILE_K
    iterations_per_tg = num_intermediate_chunks * num_k_tiles_hidden

    # Memory
    tg_memory_bytes = (
        3 * MOE_TILE_K * MOE_TILE_N * 2  # B_gate, B_up, B_down (half)
        + MOE_TILE_N * 2  # swiglu_result (half)
        + MOE_TILE_K * 2  # A_tile (half)
        + 2 * MOE_TILE_N * 2  # gate_acc_tg, up_acc_tg (half)
        + MOE_TILE_N * 2  # output_tile (half)
    )

    print(f"  Grid: ({grid_x}, {grid_y}, {grid_z}) = {total_threadgroups} threadgroups")
    print("  Threads per tg: 128")
    print(f"  Total threads: {total_threadgroups * 128}")
    print(f"  Intermediate chunks: {num_intermediate_chunks}")
    print(f"  K tiles (hidden): {num_k_tiles_hidden}")
    print(f"  Iterations per threadgroup: {iterations_per_tg}")
    print(f"  Total tile iterations: {iterations_per_tg * total_threadgroups}")
    print(f"  Threadgroup memory: {tg_memory_bytes} bytes ({tg_memory_bytes / 1024:.1f} KB)")

    # Weight buffer sizes
    num_tiles_k = (hidden_dim + TRELLIS_TILE - 1) // TRELLIS_TILE
    num_tiles_n = (intermediate_dim + TRELLIS_TILE - 1) // TRELLIS_TILE
    packed_bytes = 96  # 16*16*3/8
    expert_weights_bytes = num_tiles_k * num_tiles_n * packed_bytes
    total_weights_bytes = expert_weights_bytes * num_experts * 3  # gate, up, down

    print(f"  Expert weight size: {expert_weights_bytes / 1024:.1f} KB")
    print(f"  Total weights (all experts): {total_weights_bytes / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Debug MoE kernel scaling")
    parser.add_argument("--max-experts", type=int, default=64, help="Max experts to test")
    parser.add_argument("--real-model", action="store_true", help="Test with real model dimensions")
    parser.add_argument("--params-only", action="store_true", help="Only print dispatch parameters")
    args = parser.parse_args()

    if args.params_only:
        # Just print parameters for key configurations
        configs = [
            MoEConfig(hidden_dim=256, intermediate_dim=512, num_experts=4, top_k=2, batch_size=1),
            MoEConfig(
                hidden_dim=2048, intermediate_dim=1536, num_experts=64, top_k=8, batch_size=1
            ),
            MoEConfig(
                hidden_dim=2048, intermediate_dim=1536, num_experts=64, top_k=8, batch_size=10
            ),
        ]
        for config in configs:
            test_dispatch_parameters(config)
        return

    if args.real_model:
        # Test the exact configuration that fails
        config = MoEConfig(
            hidden_dim=2048,
            intermediate_dim=1536,
            num_experts=64,
            top_k=8,
            batch_size=1,
        )
        test_dispatch_parameters(config)
        test_single_expert(config)

        print("\n" + "=" * 80)
        print("Testing fast path...")
        result = test_moe_config(config, timeout_sec=30.0, use_fast=True)
        if result["success"]:
            print(f"Result: NaN={result['nan']}, time={result['time_ms']:.2f}ms")
        else:
            print(f"Result: FAILED - {result['error']}")
    else:
        run_scaling_tests(max_experts=args.max_experts)


if __name__ == "__main__":
    main()
