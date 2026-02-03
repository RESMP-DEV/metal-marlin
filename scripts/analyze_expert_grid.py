#!/usr/bin/env python3
"""Analyze MoE kernel grid dimensions for parallelism.

Examines whether the current kernel dispatch strategy utilizes available GPU
execution units effectively. M4 Max has ~40 execution units that can run
threadgroups in parallel.
"""

from __future__ import annotations


def select_moe_kernel(batch_size: int, use_fp32_acc: bool) -> tuple[str, int]:
    """Select optimal MoE kernel and tile size for given batch size.

    Mirrors the logic in metal_marlin/trellis/moe_dispatch.py
    """
    if batch_size == 1:
        base = "moe_trellis_swiglu_decode"
        tile_n = 32
    elif batch_size > 8:
        base = "moe_trellis_swiglu_large_batch"
        tile_n = 128
    elif batch_size >= 2:
        base = "moe_trellis_swiglu_prefill4"
        tile_n = 64
    else:
        base = "moe_trellis_swiglu"
        tile_n = 64

    if use_fp32_acc:
        if base in ("moe_trellis_swiglu_decode", "moe_trellis_swiglu_large_batch"):
            return base, tile_n
        return base + "_fp32acc", tile_n
    return base, tile_n


def compute_grid(
    batch_size: int, hidden_dim: int, top_k: int, use_fp32_acc: bool
) -> tuple[str, int, tuple[int, int, int], int]:
    """Compute grid dimensions for MoE kernel dispatch.

    Returns:
        kernel_name: Name of the kernel to dispatch
        tile_n: Tile size for output dimension
        grid: (grid_x, grid_y, grid_z) threadgroup counts
        threads_per_tg: Threads per threadgroup
    """
    kernel_name, tile_n = select_moe_kernel(batch_size, use_fp32_acc)
    is_decode_kernel = kernel_name == "moe_trellis_swiglu_decode"
    is_prefill4_kernel = "prefill4" in kernel_name

    if is_decode_kernel:
        threads_per_tg = 64
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = top_k
        grid_z = 1
    elif is_prefill4_kernel:
        threads_per_tg = 128
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = (batch_size + 3) // 4
        grid_z = top_k
    else:
        threads_per_tg = 128
        grid_x = (hidden_dim + tile_n - 1) // tile_n
        grid_y = batch_size
        grid_z = top_k

    return kernel_name, tile_n, (grid_x, grid_y, grid_z), threads_per_tg


def analyze_parallelism(
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    use_fp32_acc: bool = True,
    m4_max_exec_units: int = 40,
) -> None:
    """Analyze MoE kernel parallelism for given configuration."""
    kernel_name, tile_n, grid, threads_per_tg = compute_grid(
        batch_size, hidden_dim, top_k, use_fp32_acc
    )
    grid_x, grid_y, grid_z = grid
    total_threadgroups = grid_x * grid_y * grid_z

    print("Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  intermediate_dim: {intermediate_dim}")
    print(f"  top_k: {top_k}")
    print(f"  use_fp32_acc: {use_fp32_acc}")
    print()
    print("Kernel Selection:")
    print(f"  kernel: {kernel_name}")
    print(f"  tile_n: {tile_n}")
    print(f"  threads_per_threadgroup: {threads_per_tg}")
    print()
    print("Grid Dimensions:")
    print(f"  grid_x (output tiles): {grid_x} = ceil({hidden_dim}/{tile_n})")
    print(f"  grid_y (tokens/batches): {grid_y}")
    print(f"  grid_z (experts/slots): {grid_z}")
    print(f"  total threadgroups: {total_threadgroups}")
    print()
    print(f"Parallelism Analysis (M4 Max ~{m4_max_exec_units} execution units):")

    utilization = min(1.0, total_threadgroups / m4_max_exec_units) * 100
    print(f"  GPU utilization: {utilization:.1f}%")

    if total_threadgroups < m4_max_exec_units:
        print(f"  WARNING: Under-utilizing GPU ({total_threadgroups} < {m4_max_exec_units})")
        waves = 1
    else:
        waves = (total_threadgroups + m4_max_exec_units - 1) // m4_max_exec_units
        print("  OK: Sufficient parallel work")

    print(f"  Dispatch waves: {waves}")

    # Analyze expert parallelism specifically
    print()
    print("Expert Parallelism:")
    if "decode" in kernel_name:
        # Decode kernel: grid_y = top_k, all experts processed in parallel
        print(f"  All {top_k} experts dispatched in parallel (grid_y={grid_y})")
        print(f"  Each expert processes {grid_x} output tiles")
    elif "prefill4" in kernel_name:
        # Prefill4: grid_z = top_k
        token_batches = (batch_size + 3) // 4
        print(f"  All {top_k} experts dispatched in parallel (grid_z={grid_z})")
        print(f"  Processing {token_batches} token batches of 4")
        print(f"  Each expert-batch pair processes {grid_x} output tiles")
    else:
        # Generic: grid_z = top_k
        print(f"  All {top_k} experts dispatched in parallel (grid_z={grid_z})")
        print(f"  Each expert processes {batch_size} tokens x {grid_x} output tiles")


def main() -> None:
    """Analyze grid dimensions for typical MoE configurations."""
    # DeepSeek-style config
    configs = [
        # (batch_size, hidden_dim, intermediate_dim, top_k, description)
        (1, 2048, 1536, 8, "Decode (batch=1)"),
        (1, 5120, 1536, 8, "Decode larger hidden"),
        (4, 2048, 1536, 8, "Prefill batch=4"),
        (8, 2048, 1536, 8, "Prefill batch=8"),
        (16, 2048, 1536, 8, "Large batch=16"),
        (32, 2048, 1536, 8, "Large batch=32"),
        # MiniMax/Larger models
        (1, 4096, 2048, 8, "Decode MiniMax-style"),
        (1, 4096, 2048, 6, "Decode top_k=6"),
    ]

    for batch_size, hidden_dim, intermediate_dim, top_k, desc in configs:
        print("=" * 70)
        print(f"  {desc}")
        print("=" * 70)
        analyze_parallelism(batch_size, hidden_dim, intermediate_dim, top_k)
        print()


if __name__ == "__main__":
    main()
