#!/usr/bin/env python3
"""Focused benchmark for MR-GPTQ backend timings.

Compares backend timings for representative GLM-4.7-style and
Qwen3-Coder-Next-like matrix shapes.

Measures:
- Hessian build time
- Cholesky/inversion time
- Quantization loop time

Usage:
  cd contrib/metal_marlin
  uv run python benchmarks/bench_mr_gptq_backends.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metal_marlin.gptq_accelerated import GPTQAccelerated, GPTQConfig

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "mr_gptq_backend_baseline.json"
BENCHMARK_NAME = "mr_gptq_backend_baseline"


@dataclass(frozen=True)
class WorkloadPreset:
    """Benchmark workload configuration."""

    name: str
    description: str
    out_features: int
    in_features: int
    calibration_tokens: int
    group_size: int = 128


# Scaled proxy shapes that preserve common MoE/MLP aspect ratios while
# keeping runtime practical for repeated local benchmark runs.
WORKLOAD_PRESETS: dict[str, WorkloadPreset] = {
    "glm47_moe": WorkloadPreset(
        name="glm47_moe",
        description="GLM-4.7-style MoE FFN proxy",
        out_features=4096,
        in_features=1024,
        calibration_tokens=2048,
    ),
    "qwen3_coder_next": WorkloadPreset(
        name="qwen3_coder_next",
        description="Qwen3-Coder-Next-style FFN proxy",
        out_features=6144,
        in_features=1536,
        calibration_tokens=2048,
    ),
}


@dataclass
class BenchmarkRow:
    """Serialized benchmark row for one workload/backend pair."""

    workloads: str
    backend: str
    hessian_ms: float
    cholesky_ms: float
    quantize_ms: float
    timestamp: str
    out_features: int
    in_features: int
    calibration_tokens: int
    runs: int
    warmup: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _phase_map(rows: list[BenchmarkRow], phase: str) -> dict[str, float]:
    return {f"{row.workloads}:{row.backend}": getattr(row, phase) for row in rows}


def _available_backends() -> set[str]:
    backends = {"numpy"}
    try:
        import torch
import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

    except Exception:
        return backends

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.add("mps")
    if torch.cuda.is_available():
        backends.add("cuda")
    return backends


def _resolve_backends(requested: list[str]) -> list[str]:
    normalized = [name.lower() for name in requested]
    available = _available_backends()
    resolved: list[str] = []

    for backend in normalized:
        if backend == "auto":
            for candidate in ("numpy", "mps", "cuda"):
                if candidate in available and candidate not in resolved:
                    resolved.append(candidate)
            continue

        if backend not in {"numpy", "mps", "cuda"}:
            raise ValueError(
                f"Unknown backend '{backend}'. Valid options: auto, numpy, mps, cuda."
            )

        if backend in available and backend not in resolved:
            resolved.append(backend)

    return resolved


def _make_inputs(workload: WorkloadPreset, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((workload.out_features, workload.in_features)).astype(
        np.float32
    )
    activations = rng.standard_normal(
        (workload.calibration_tokens, workload.in_features)
    ).astype(np.float32)
    return weights, activations


def run_workload_backend(
    workload: WorkloadPreset,
    backend: str,
    runs: int,
    warmup: int,
    seed: int,
) -> BenchmarkRow | None:
    weights, activations = _make_inputs(workload, seed=seed)
    config = GPTQConfig(group_size=workload.group_size, actorder=True, damp=0.01)

    try:
        quantizer = GPTQAccelerated.create(backend=backend, config=config)
    except Exception as exc:
        print(f"[skip] backend={backend} workload={workload.name}: {exc}")
        return None

    hessian_ms: list[float] = []
    cholesky_ms: list[float] = []
    quantize_ms: list[float] = []
    actual_backend = backend

    total_iters = warmup + runs
    for i in range(total_iters):
        try:
            result = quantizer.quantize_layer(
                layer_name=f"{workload.name}.synthetic",
                weights=weights,
                activations=activations,
            )
        except Exception as exc:
            print(f"[skip] backend={backend} workload={workload.name}: {exc}")
            return None
        actual_backend = result.backend

        if i >= warmup:
            hessian_ms.append(result.time_hessian * 1000.0)
            cholesky_ms.append(result.time_cholesky * 1000.0)
            quantize_ms.append(result.time_quantize * 1000.0)

    return BenchmarkRow(
        workloads=workload.name,
        backend=actual_backend,
        hessian_ms=statistics.mean(hessian_ms),
        cholesky_ms=statistics.mean(cholesky_ms),
        quantize_ms=statistics.mean(quantize_ms),
        timestamp=_utc_timestamp(),
        out_features=workload.out_features,
        in_features=workload.in_features,
        calibration_tokens=workload.calibration_tokens,
        runs=runs,
        warmup=warmup,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MR-GPTQ backend timings.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=sorted(WORKLOAD_PRESETS.keys()),
        default=["glm47_moe", "qwen3_coder_next"],
        help="Workload presets to benchmark.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["auto"],
        help="Backends: auto, numpy, mps, cuda.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per workload/backend pair.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per workload/backend pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260207,
        help="Random seed for synthetic input generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    selected_backends = _resolve_backends(args.backends)
    if not selected_backends:
        print("No requested backends are available on this machine.")
        return 1

    print("=" * 72)
    print("MR-GPTQ backend benchmark")
    print("=" * 72)
    print(f"Workloads: {', '.join(args.workloads)}")
    print(f"Backends: {', '.join(selected_backends)}")
    print(f"Runs: {args.runs} (warmup: {args.warmup})")
    print("-" * 72)

    rows: list[BenchmarkRow] = []
    for workload_name in args.workloads:
        workload = WORKLOAD_PRESETS[workload_name]
        print(f"Workload: {workload.name} ({workload.description})")
        print(
            f"  shape=[{workload.out_features}, {workload.in_features}], "
            f"calibration_tokens={workload.calibration_tokens}"
        )

        for backend in selected_backends:
            seed_offset = sum(ord(ch) for ch in f"{workload_name}:{backend}")
            row = run_workload_backend(
                workload=workload,
                backend=backend,
                runs=args.runs,
                warmup=args.warmup,
                seed=args.seed + seed_offset,
            )
            if row is None:
                continue
            rows.append(row)
            print(
                f"  {row.backend:<8} hessian={row.hessian_ms:8.2f} ms  "
                f"cholesky={row.cholesky_ms:8.2f} ms  "
                f"quantize={row.quantize_ms:8.2f} ms"
            )

    if not rows:
        print("No benchmark runs completed successfully.")
        return 1

    payload = {
        "benchmark": BENCHMARK_NAME,
        "workloads": [
            {
                "name": WORKLOAD_PRESETS[name].name,
                "description": WORKLOAD_PRESETS[name].description,
                "out_features": WORKLOAD_PRESETS[name].out_features,
                "in_features": WORKLOAD_PRESETS[name].in_features,
                "calibration_tokens": WORKLOAD_PRESETS[name].calibration_tokens,
            }
            for name in args.workloads
        ],
        "backend": selected_backends,
        "hessian_ms": _phase_map(rows, "hessian_ms"),
        "cholesky_ms": _phase_map(rows, "cholesky_ms"),
        "quantize_ms": _phase_map(rows, "quantize_ms"),
        "timestamp": _utc_timestamp(),
        "results": [asdict(row) for row in rows],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Wrote results to: {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
