"""Auto-tuning framework for selecting the best Metal Marlin kernel config."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

from .dtypes import get_default_config
from .kernels import pack_fp4_weights

if TYPE_CHECKING:
    from .dtypes import DTypeConfig

DEFAULT_GROUP_SIZE = 32
_KERNEL_SOURCE_PATH = Path(__file__).parent.parent / "src" / "kernels_autotune.metal"
_kernel_source: str | None = None
_kernel_cache: dict[str, object] = {}


@dataclass
class KernelConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    num_stages: int
    threadgroup_size: tuple[int, int, int]
    kernel_name: str


CONFIGURATIONS = [
    KernelConfig(32, 32, 16, 2, (128, 1, 1), "marlin_gemm_fp4_32x32x16"),
    KernelConfig(64, 64, 32, 2, (256, 1, 1), "marlin_gemm_fp4_64x64x32"),
    KernelConfig(128, 64, 32, 3, (512, 1, 1), "marlin_gemm_fp4_128x64x32"),
]

# Cache of (M, N, K) -> best config
_autotune_cache: dict[tuple[int, int, int], KernelConfig] = {}


def _get_kernel_source() -> str:
    global _kernel_source
    if _kernel_source is None:
        _kernel_source = _KERNEL_SOURCE_PATH.read_text()
    return _kernel_source


def _resolve_kernel_name(config: KernelConfig) -> str:
    source = _get_kernel_source()
    if config.kernel_name in source:
        return config.kernel_name
    fallback = f"marlin_gemm_fp4_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
    if fallback in source:
        return fallback
    raise ValueError(f"Kernel {config.kernel_name} not found in {_KERNEL_SOURCE_PATH}")


def _get_kernel(kernel_name: str) -> object:
    if kernel_name not in _kernel_cache:
        kernel = mx.fast.metal_kernel(
            name=kernel_name,
            input_names=["A", "B", "scales"],
            output_names=["C"],
            source=_get_kernel_source(),
            ensure_row_contiguous=True,
        )
        _kernel_cache[kernel_name] = kernel
    return _kernel_cache[kernel_name]


def run_kernel(
    config: KernelConfig,
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = DEFAULT_GROUP_SIZE,
    dtype_config: DTypeConfig | None = None,
) -> mx.array:
    """Dispatch a specific kernel variant for A @ B."""
    M, K = A.shape
    N = B_packed.shape[1]
    grid = (
        (N + config.tile_n - 1) // config.tile_n,
        (M + config.tile_m - 1) // config.tile_m,
        1,
    )
    kernel_name = _resolve_kernel_name(config)
    kernel = _get_kernel(kernel_name)

    # Use provided dtype config or default (BF16)
    cfg = dtype_config if dtype_config is not None else get_default_config()
    output_dtype = cfg.mlx_activations

    outputs = kernel(
        inputs=[A, B_packed, scales],
        template=[("M", M), ("N", N), ("K", K), ("group_size", group_size)],
        grid=grid,
        threadgroup=config.threadgroup_size,
        output_shapes=[(M, N)],
        output_dtypes=[output_dtype],
        init_value=0.0,
    )
    return outputs[0]


def autotune_gemm(
    M: int,
    N: int,
    K: int,
    warmup: int = 3,
    iters: int = 10,
    dtype_config: DTypeConfig | None = None,
) -> KernelConfig:
    """Find fastest kernel configuration for given problem size."""
    cache_key = (M, N, K)
    if cache_key in _autotune_cache:
        return _autotune_cache[cache_key]

    # Use provided dtype config or default (BF16)
    cfg = dtype_config if dtype_config is not None else get_default_config()
    act_dtype = cfg.mlx_activations

    best_config = None
    best_time = float("inf")

    A = mx.random.normal((M, K), dtype=act_dtype)
    weight = mx.random.normal((N, K), dtype=act_dtype)
    packed = pack_fp4_weights(weight, group_size=DEFAULT_GROUP_SIZE)
    if len(packed) == 3:
        B_packed, scales, _meta = packed
    else:
        B_packed, scales = packed
    mx.eval(A, B_packed, scales)

    for config in CONFIGURATIONS:
        # Skip configs that don't fit
        if M % config.tile_m != 0 or N % config.tile_n != 0:
            continue

        # Warmup
        for _ in range(warmup):
            out = run_kernel(config, A, B_packed, scales, dtype_config=cfg)
            mx.eval(out)
            mx.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            out = run_kernel(config, A, B_packed, scales, dtype_config=cfg)
            mx.eval(out)
            mx.synchronize()
        elapsed = (time.perf_counter() - start) / iters

        if elapsed < best_time:
            best_time = elapsed
            best_config = config

    if best_config is None:
        best_config = CONFIGURATIONS[0]

    _autotune_cache[cache_key] = best_config
    return best_config


def autotuned_linear(
    x: mx.array,
    weight_packed: mx.array,
    scales: mx.array,
    group_size: int = DEFAULT_GROUP_SIZE,
    warmup: int = 3,
    iters: int = 10,
    dtype_config: DTypeConfig | None = None,
) -> mx.array:
    """Quantized linear with automatic kernel selection."""
    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d
    N = weight_packed.shape[1]

    config = autotune_gemm(M, N, K, warmup=warmup, iters=iters, dtype_config=dtype_config)
    return run_kernel(config, x, weight_packed, scales, group_size=group_size, dtype_config=dtype_config)


def _config_to_dict(config: KernelConfig) -> dict[str, object]:
    return {
        "tile_m": config.tile_m,
        "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "num_stages": config.num_stages,
        "threadgroup_size": list(config.threadgroup_size),
        "kernel_name": config.kernel_name,
    }


def _config_from_dict(data: dict[str, object]) -> KernelConfig:
    return KernelConfig(
        tile_m=int(data["tile_m"]),
        tile_n=int(data["tile_n"]),
        tile_k=int(data["tile_k"]),
        num_stages=int(data["num_stages"]),
        threadgroup_size=tuple(data["threadgroup_size"]),
        kernel_name=str(data["kernel_name"]),
    )


def save_autotune_cache(path: str) -> None:
    """Save cache for production use without re-tuning."""
    output = {
        "version": 1,
        "entries": [
            {"M": m, "N": n, "K": k, "config": _config_to_dict(cfg)}
            for (m, n, k), cfg in _autotune_cache.items()
        ],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))


def load_autotune_cache(path: str) -> None:
    """Load pre-computed cache."""
    in_path = Path(path)
    if not in_path.exists():
        return

    data = json.loads(in_path.read_text())
    if data.get("version") != 1:
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
    """Run autotune across a list of (M, N, K) shapes."""
    results = []
    for M, N, K in sizes:
        results.append(autotune_gemm(M, N, K, warmup=warmup, iters=iters, dtype_config=dtype_config))
    return results
