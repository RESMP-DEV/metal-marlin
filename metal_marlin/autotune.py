"""Auto-tuning framework for selecting the best Metal Marlin kernel config.

Benchmarks multiple kernel tile configurations and selects the fastest for each
problem size (M, N, K). Uses direct Metal kernel dispatch via PyObjC without MLX.

Usage:
    from metal_marlin.autotune import autotune_gemm, autotuned_linear

    # Find best kernel for given shape
    config = autotune_gemm(M=256, N=4096, K=4096)

    # Use auto-tuned kernel for linear layer
    output = autotuned_linear(x, weight_packed, scales)

    # Save/load tuning results for production
    save_autotune_cache("tune_results.json")
    load_autotune_cache("tune_results.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import HAS_MPS, HAS_TORCH, require_torch, torch
from .metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_mps,
)

if TYPE_CHECKING:
    from .dtypes import DTypeConfig

DEFAULT_GROUP_SIZE = 32
_KERNEL_SOURCE_PATH = Path(__file__).parent.parent / "src" / "kernels_autotune.metal"


@dataclass
class KernelConfig:
    """Configuration for a specific kernel tile variant."""

    tile_m: int
    tile_n: int
    tile_k: int
    num_stages: int
    threadgroup_size: tuple[int, int, int]
    kernel_name: str


# Kernel configurations matching kernels_autotune.metal variants
CONFIGURATIONS = [
    # Small tiles - high occupancy, good for small M
    KernelConfig(32, 32, 32, 2, (128, 1, 1), "marlin_gemm_fp4_t32x32x32"),
    KernelConfig(32, 32, 16, 2, (128, 1, 1), "marlin_gemm_fp4_32x32x16"),
    # Asymmetric tiles - for M != N shapes
    KernelConfig(64, 32, 32, 2, (128, 1, 1), "marlin_gemm_fp4_t64x32x32"),
    KernelConfig(32, 64, 32, 2, (128, 1, 1), "marlin_gemm_fp4_t32x64x32"),
    # Default balanced config
    KernelConfig(64, 64, 32, 2, (128, 1, 1), "marlin_gemm_fp4_t64x64x32"),
    KernelConfig(64, 64, 32, 2, (128, 1, 1), "marlin_gemm_fp4_64x64x32"),
    # Shallow K - for memory-bound shapes
    KernelConfig(64, 64, 16, 2, (128, 1, 1), "marlin_gemm_fp4_t64x64x16"),
    # Deep K - for compute-bound shapes
    KernelConfig(64, 64, 64, 2, (128, 1, 1), "marlin_gemm_fp4_t64x64x64"),
    # Large tiles - for large M or N
    KernelConfig(128, 64, 32, 3, (256, 1, 1), "marlin_gemm_fp4_t128x64x32"),
    KernelConfig(128, 64, 32, 3, (256, 1, 1), "marlin_gemm_fp4_128x64x32"),
    KernelConfig(64, 128, 32, 3, (256, 1, 1), "marlin_gemm_fp4_t64x128x32"),
    KernelConfig(64, 128, 32, 3, (256, 1, 1), "marlin_gemm_fp4_64x128x32"),
    # Maximum tile
    KernelConfig(128, 128, 16, 3, (256, 1, 1), "marlin_gemm_fp4_t128x128x16"),
]

# Cache of (M, N, K) -> best config
_autotune_cache: dict[tuple[int, int, int], KernelConfig] = {}

# Cached kernel library
_kernel_library: MetalKernelLibrary | None = None


def _get_kernel_library() -> MetalKernelLibrary:
    """Get or create the kernel library for autotuning."""
    global _kernel_library
    if _kernel_library is None:
        _kernel_library = get_default_library()
    return _kernel_library


def _resolve_kernel_name(config: KernelConfig, lib: MetalKernelLibrary) -> str:
    """Resolve kernel name to an available variant in the library.

    Args:
        config: Kernel configuration to resolve.
        lib: Metal kernel library with compiled shaders.

    Returns:
        Resolved kernel name.

    Raises:
        ValueError: If no matching kernel is found.
    """
    # Try the exact kernel name first
    try:
        lib.get_pipeline(config.kernel_name)
        return config.kernel_name
    except KeyError:
        pass

    # Try alternative naming conventions
    alternatives = [
        f"marlin_gemm_fp4_t{config.tile_m}x{config.tile_n}x{config.tile_k}",
        f"marlin_gemm_fp4_{config.tile_m}x{config.tile_n}x{config.tile_k}",
    ]

    for name in alternatives:
        try:
            lib.get_pipeline(name)
            return name
        except KeyError:
            continue

    raise ValueError(
        f"Kernel {config.kernel_name} not found in library. "
        f"Tried: {[config.kernel_name] + alternatives}"
    )


def _pack_fp4_weights_torch(
    weight: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack FP16/BF16 weights to FP4 E2M1 format using PyTorch.

    Args:
        weight: Weight tensor [N, K] in FP16/BF16.
        group_size: Number of weights per scale group.

    Returns:
        Tuple of (packed weights [K//8, N], scales [K//group_size, N]).
    """
    require_torch("weight packing")

    N, K = weight.shape
    device = weight.device

    # Compute per-group scales (absmax scaling)
    weight_groups = weight.view(N, K // group_size, group_size)
    scales = weight_groups.abs().amax(dim=-1)  # [N, K//group_size]
    scales = scales.clamp(min=1e-8)  # Avoid division by zero

    # Normalize weights by scale
    scales_expanded = scales.unsqueeze(-1).expand(-1, -1, group_size)
    scales_expanded = scales_expanded.reshape(N, K)
    normalized = weight / scales_expanded

    # Quantize to FP4 E2M1 (values in [-6, 6] range)
    # FP4 E2M1 representable values: 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    # Clamp to representable range
    normalized = normalized.clamp(-6.0, 6.0)

    # Convert to nibbles (simplified quantization)
    # Map to 16 levels: sign (1 bit) + exponent (2 bits) + mantissa (1 bit)
    sign = (normalized < 0).int()
    abs_val = normalized.abs()

    # Encode to FP4 nibble
    nibbles = torch.zeros_like(abs_val, dtype=torch.int32)

    # exp=0: subnormal, values 0, 0.25
    mask_sub = abs_val < 0.375
    nibbles = torch.where(mask_sub & (abs_val >= 0.125), torch.ones_like(nibbles), nibbles)

    # exp=1: values 0.5, 0.75
    mask_e1 = (abs_val >= 0.375) & (abs_val < 1.25)
    nibbles = torch.where(mask_e1 & (abs_val < 0.625), 2 * torch.ones_like(nibbles), nibbles)
    nibbles = torch.where(mask_e1 & (abs_val >= 0.625), 3 * torch.ones_like(nibbles), nibbles)

    # exp=2: values 1.0, 1.5
    mask_e2 = (abs_val >= 1.25) & (abs_val < 2.5)
    nibbles = torch.where(mask_e2 & (abs_val < 1.75), 4 * torch.ones_like(nibbles), nibbles)
    nibbles = torch.where(mask_e2 & (abs_val >= 1.75), 5 * torch.ones_like(nibbles), nibbles)

    # exp=3: values 2.0, 3.0, 4.0, 6.0
    mask_e3 = abs_val >= 2.5
    nibbles = torch.where(mask_e3 & (abs_val < 3.5), 6 * torch.ones_like(nibbles), nibbles)
    nibbles = torch.where(mask_e3 & (abs_val >= 3.5), 7 * torch.ones_like(nibbles), nibbles)

    # Add sign bit
    nibbles = nibbles | (sign << 3)

    # Pack 8 nibbles per uint32, layout [K//8, N]
    nibbles = nibbles.view(N, K // 8, 8).permute(1, 0, 2)  # [K//8, N, 8]
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed = packed | (nibbles[:, :, i] << (i * 4))

    # Transpose scales to [K//group_size, N]
    scales = scales.T.contiguous()

    return packed.to(torch.int32), scales.half()


def run_kernel(
    config: KernelConfig,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    dtype_config: DTypeConfig | None = None,
    lib: MetalKernelLibrary | None = None,
) -> torch.Tensor:
    """Dispatch a specific kernel variant for A @ B.

    Args:
        config: Kernel configuration specifying tile sizes.
        A: Input activations [M, K], fp16/bf16, MPS tensor.
        B_packed: Packed FP4 weights [K//8, N], uint32, MPS.
        scales: Per-group scales [K//group_size, N], fp16, MPS.
        group_size: Quantization group size.
        dtype_config: Optional dtype configuration.
        lib: Optional kernel library (uses default if None).

    Returns:
        Output tensor [M, N], fp16, MPS.
    """
    require_mps()

    if lib is None:
        lib = _get_kernel_library()

    M, K = A.shape
    # B_packed is [K//8, N]
    N = B_packed.shape[1]

    # Resolve kernel name
    kernel_name = _resolve_kernel_name(config, lib)

    # Compute grid dimensions
    grid_m = (M + config.tile_m - 1) // config.tile_m
    grid_n = (N + config.tile_n - 1) // config.tile_n

    # Allocate output
    output = torch.empty((M, N), dtype=torch.float16, device="mps")

    # Create Metal buffers
    device = lib.device
    A_buf = mps_tensor_to_metal_buffer(A.half().contiguous(), device)
    B_buf = mps_tensor_to_metal_buffer(B_packed.contiguous(), device)
    S_buf = mps_tensor_to_metal_buffer(scales.half().contiguous(), device)
    C_buf = mps_tensor_to_metal_buffer(output, device)

    # Create params buffer: M, N, K, group_size
    import Metal

    M_buf = device.newBufferWithBytes_length_options_(
        np.array([M], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    N_buf = device.newBufferWithBytes_length_options_(
        np.array([N], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    K_buf = device.newBufferWithBytes_length_options_(
        np.array([K], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )

    # Dispatch kernel
    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_n, grid_m, 1),  # Note: grid is (N tiles, M tiles) for column-major output
        threadgroup=config.threadgroup_size,
        buffers=[A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf],
        wait=True,
    )

    return output


def benchmark_kernel(
    config: KernelConfig,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    warmup: int = 3,
    iters: int = 10,
    lib: MetalKernelLibrary | None = None,
) -> float:
    """Benchmark a single kernel configuration.

    Args:
        config: Kernel configuration to benchmark.
        A: Input activations [M, K], MPS tensor.
        B_packed: Packed weights [K//8, N], MPS tensor.
        scales: Scales [K//group_size, N], MPS tensor.
        group_size: Quantization group size.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.
        lib: Optional kernel library.

    Returns:
        Average execution time in seconds.
    """
    require_mps()

    if lib is None:
        lib = _get_kernel_library()

    # Warmup
    for _ in range(warmup):
        _ = run_kernel(config, A, B_packed, scales, group_size, lib=lib)
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = run_kernel(config, A, B_packed, scales, group_size, lib=lib)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters


def autotune_gemm(
    M: int,
    N: int,
    K: int,
    warmup: int = 3,
    iters: int = 10,
    dtype_config: DTypeConfig | None = None,
) -> KernelConfig:
    """Find fastest kernel configuration for given problem size.

    Benchmarks all compatible kernel configurations and returns the fastest.
    Results are cached for subsequent calls with the same dimensions.

    Args:
        M: Number of rows in output (batch * seq for transformers).
        N: Number of columns in output (output features).
        K: Inner dimension (input features).
        warmup: Number of warmup iterations per config.
        iters: Number of benchmark iterations per config.
        dtype_config: Optional dtype configuration.

    Returns:
        Best performing KernelConfig for this problem size.
    """
    require_mps()

    cache_key = (M, N, K)
    if cache_key in _autotune_cache:
        return _autotune_cache[cache_key]

    lib = _get_kernel_library()

    # Create test data
    A = torch.randn(M, K, dtype=torch.float16, device="mps")
    weight = torch.randn(N, K, dtype=torch.float16, device="mps")
    B_packed, scales = _pack_fp4_weights_torch(weight, group_size=DEFAULT_GROUP_SIZE)

    best_config = None
    best_time = float("inf")

    for config in CONFIGURATIONS:
        # Check if kernel is available
        try:
            _resolve_kernel_name(config, lib)
        except ValueError:
            continue

        try:
            elapsed = benchmark_kernel(
                config,
                A,
                B_packed,
                scales,
                group_size=DEFAULT_GROUP_SIZE,
                warmup=warmup,
                iters=iters,
                lib=lib,
            )

            if elapsed < best_time:
                best_time = elapsed
                best_config = config

        except Exception:
            # Kernel dispatch failed, skip this config
            continue

    if best_config is None:
        # Fallback to default config
        best_config = CONFIGURATIONS[4]  # 64x64x32 default

    _autotune_cache[cache_key] = best_config
    return best_config


def autotuned_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    warmup: int = 3,
    iters: int = 10,
    dtype_config: DTypeConfig | None = None,
) -> torch.Tensor:
    """Quantized linear with automatic kernel selection.

    Flattens input to 2D, runs autotuned GEMM, and reshapes output.

    Args:
        x: Input tensor [..., K].
        weight_packed: Packed FP4 weights [K//8, N].
        scales: Per-group scales [K//group_size, N].
        group_size: Quantization group size.
        warmup: Warmup iterations for autotuning.
        iters: Benchmark iterations for autotuning.
        dtype_config: Optional dtype configuration.

    Returns:
        Output tensor [..., N].
    """
    require_mps()

    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d
    N = weight_packed.shape[1]

    # Reshape to 2D
    x_2d = x.view(M, K)

    # Get best config
    config = autotune_gemm(M, N, K, warmup=warmup, iters=iters, dtype_config=dtype_config)

    # Run kernel
    output = run_kernel(
        config, x_2d, weight_packed, scales, group_size=group_size, dtype_config=dtype_config
    )

    # Reshape output
    out_shape = orig_shape[:-1] + (N,)
    return output.view(out_shape)


def _config_to_dict(config: KernelConfig) -> dict[str, object]:
    """Serialize KernelConfig to dict for JSON storage."""
    return {
        "tile_m": config.tile_m,
        "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "num_stages": config.num_stages,
        "threadgroup_size": list(config.threadgroup_size),
        "kernel_name": config.kernel_name,
    }


def _config_from_dict(data: dict[str, object]) -> KernelConfig:
    """Deserialize KernelConfig from dict."""
    return KernelConfig(
        tile_m=int(data["tile_m"]),
        tile_n=int(data["tile_n"]),
        tile_k=int(data["tile_k"]),
        num_stages=int(data["num_stages"]),
        threadgroup_size=tuple(data["threadgroup_size"]),
        kernel_name=str(data["kernel_name"]),
    )


def save_autotune_cache(path: str) -> None:
    """Save cache for production use without re-tuning.

    Args:
        path: Output file path for JSON cache.
    """
    output = {
        "version": 2,  # Version 2: PyTorch/Metal dispatch (no MLX)
        "entries": [
            {"M": m, "N": n, "K": k, "config": _config_to_dict(cfg)}
            for (m, n, k), cfg in _autotune_cache.items()
        ],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))


def load_autotune_cache(path: str) -> None:
    """Load pre-computed cache.

    Args:
        path: Input file path for JSON cache.
    """
    in_path = Path(path)
    if not in_path.exists():
        return

    data = json.loads(in_path.read_text())
    version = data.get("version", 1)

    # Accept both version 1 (MLX) and version 2 (PyTorch) caches
    if version not in (1, 2):
        return

    for entry in data.get("entries", []):
        key = (int(entry["M"]), int(entry["N"]), int(entry["K"]))
        _autotune_cache[key] = _config_from_dict(entry["config"])


def sweep_problem_sizes(
    sizes: list[tuple[int, int, int]],
    warmup: int = 3,
    iters: int = 10,
    dtype_config: DTypeConfig | None = None,
) -> list[KernelConfig]:
    """Run autotune across a list of (M, N, K) shapes.

    Args:
        sizes: List of problem sizes to tune.
        warmup: Warmup iterations per config.
        iters: Benchmark iterations per config.
        dtype_config: Optional dtype configuration.

    Returns:
        List of best KernelConfig for each size.
    """
    results = []
    for M, N, K in sizes:
        results.append(
            autotune_gemm(M, N, K, warmup=warmup, iters=iters, dtype_config=dtype_config)
        )
    return results


def get_tuning_stats() -> dict[str, Any]:
    """Get statistics about the current autotuning cache.

    Returns:
        Dictionary with cache statistics.
    """
    if not _autotune_cache:
        return {"entries": 0, "configs_used": set()}

    configs_used = set()
    for cfg in _autotune_cache.values():
        configs_used.add(cfg.kernel_name)

    return {
        "entries": len(_autotune_cache),
        "configs_used": list(configs_used),
        "problem_sizes": list(_autotune_cache.keys()),
    }


def clear_cache() -> None:
    """Clear the autotuning cache."""
    _autotune_cache.clear()


# ---------------------------------------------------------------------------
# CLI / test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Metal available: {HAS_METAL}")
    print(f"MPS available: {HAS_MPS}")
    print(f"Torch available: {HAS_TORCH}")

    if HAS_MPS:
        print("\n--- Running autotune benchmark ---")

        # Test problem sizes
        sizes = [
            (1, 4096, 4096),  # Single token decode
            (32, 4096, 4096),  # Small batch
            (128, 4096, 4096),  # Medium batch
            (256, 4096, 14336),  # Large N (MLP up-projection)
            (256, 14336, 4096),  # Large input (MLP down-projection)
        ]

        for M, N, K in sizes:
            config = autotune_gemm(M, N, K, warmup=5, iters=20)
            print(
                f"  ({M}, {N}, {K}): {config.kernel_name} "
                f"[{config.tile_m}x{config.tile_n}x{config.tile_k}]"
            )

        print(f"\nCache stats: {get_tuning_stats()}")
