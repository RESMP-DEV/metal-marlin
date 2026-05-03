#!/usr/bin/env python3
"""Reproducible benchmark script for metal_marlin.

Generates machine-readable JSON results to prevent fabricated benchmark claims.
Each run captures:
  - Full hardware/software environment for reproducibility
  - Raw per-iteration timings (not just averages)
  - Statistical summary (mean, median, p95, stddev)
  - Integrity hash of the JSON content
  - Git repository state (commit, dirty files)
  - Benchmark configuration used

Output: contrib/metal_marlin/reports/bench_YYYYMMDD_HHMMSS.json

Usage:
    # From repo root:
    uv run python contrib/metal_marlin/scripts/repro_benchmark.py

    # Quick mode (fewer iterations, for CI):
    uv run python contrib/metal_marlin/scripts/repro_benchmark.py --quick

    # With tag for later identification:
    uv run python contrib/metal_marlin/scripts/repro_benchmark.py --tag baseline-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# ── Project root detection ───────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # contrib/metal_marlin/
_REPORTS_DIR = _PROJECT_ROOT / "reports"

# Ensure metal_marlin is importable
sys.path.insert(0, str(_PROJECT_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute percentile from already-sorted list."""
    logger.debug("_percentile called with sorted_vals=%s, pct=%s", sorted_vals, pct)
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _git_info() -> dict[str, Any]:
    """Capture git repository state for reproducibility."""
    logger.debug("_git_info called")
    info: dict[str, Any] = {
        "available": False,
        "commit": None,
        "branch": None,
        "dirty_files": None,
        "describe": None,
    }
    try:
        def _run(cmd: list[str]) -> str:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(_PROJECT_ROOT),
                timeout=10,
            )
            return result.stdout.strip()

        info["commit"] = _run(["git", "rev-parse", "HEAD"])
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["describe"] = _run(["git", "describe", "--always", "--dirty"])
        info["dirty_files"] = _run(["git", "status", "--porcelain"]).splitlines()
        info["available"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


def _hardware_info() -> dict[str, Any]:
    """Capture hardware environment."""
    logger.debug("_hardware_info called")
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }

    # macOS-specific: sysctl for Apple Silicon details
    if sys.platform == "darwin":
        try:
            def _sysctl(key: str) -> str:
                logger.debug("_sysctl called with key=%s", key)
                r = subprocess.run(
                    ["sysctl", "-n", key],
                    capture_output=True, text=True, timeout=5,
                )
                return r.stdout.strip()

            info["apple_chip"] = _sysctl("machdep.cpu.brand_string")
            info["physical_cpus"] = int(_sysctl("hw.physicalcpu"))
            info["logical_cpus"] = int(_sysctl("hw.logicalcpu"))
            mem_bytes = int(_sysctl("hw.memsize"))
            info["total_memory_gb"] = round(mem_bytes / (1024 ** 3), 2)
        except Exception:
            pass

    # Try to get GPU info via torch
    info["gpu"] = None
    try:
        import torch  # noqa: E402
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu"] = {
                "backend": "mps",
                "torch_version": torch.__version__,
                "mps_built": torch.backends.mps.is_built(),
            }
            # Try to get Metal device info
            try:
                import Metal  # type: ignore[import-not-found]
                device = Metal.MTLCreateSystemDefaultDevice()
                info["gpu"]["device_name"] = str(device.name())
                info["gpu"]["registry_id"] = str(device.registryID())
            except Exception:
                pass
        elif torch.cuda.is_available():
            info["gpu"] = {
                "backend": "cuda",
                "torch_version": torch.__version__,
                "device_name": torch.cuda.get_device_name(0),
            }
    except ImportError:
        pass

    return info


def _software_versions() -> dict[str, Any]:
    """Capture versions of key dependencies."""
    logger.debug("_software_versions called")
    versions: dict[str, Any] = {}

    for pkg_name in ("numpy", "scipy", "psutil", "pydantic", "safetensors", "tiktoken"):
        try:
            mod = __import__(pkg_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg_name] = None

    try:
        import torch
        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda
        versions["torch_mps"] = (
            str(torch.backends.mps.is_available())
            if hasattr(torch.backends, "mps")
            else None
        )
    except ImportError:
        versions["torch"] = None

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = None

    try:
        import metal_marlin
        versions["metal_marlin"] = getattr(metal_marlin, "__version__", "0.0.0")
    except ImportError:
        versions["metal_marlin"] = "import_failed"

    return versions


def _compute_integrity_hash(data: dict[str, Any]) -> str:
    """Compute SHA-256 of the benchmark data (excluding the hash field itself).

    This allows consumers to verify the JSON was not tampered with after
    generation.  The hash covers the canonical JSON serialization of every
    key except 'integrity_hash' itself.
    """
    logger.debug("_compute_integrity_hash called with data=%s", data)
    canonical = json.dumps(data, sort_keys=True, indent=None, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Benchmark suites ─────────────────────────────────────────────────────────

def bench_numpy_matmul(
    sizes: list[tuple[int, int, int]],
    warmup: int = 3,
    iterations: int = 20,
) -> list[dict[str, Any]]:
    """Benchmark numpy matrix multiplication for baseline throughput."""
    logger.info("bench_numpy_matmul starting with sizes=%s, warmup=%s, iterations=%s", sizes, warmup, iterations)
    import numpy as np

    results: list[dict[str, Any]] = []
    for M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            _ = A @ B

        raw_times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = A @ B
            elapsed = time.perf_counter() - start
            raw_times.append(elapsed)

        sorted_t = sorted(raw_times)
        mean_s = statistics.mean(raw_times)
        ops = 2.0 * M * K * N  # FLOPs for GEMM
        gflops = ops / (mean_s * 1e9) if mean_s > 0 else 0.0

        results.append({
            "name": "numpy_matmul_fp32",
            "dimensions": {"M": M, "K": K, "N": N},
            "flops": ops,
            "warmup": warmup,
            "iterations": iterations,
            "raw_times_s": raw_times,
            "summary": {
                "mean_s": mean_s,
                "median_s": statistics.median(raw_times),
                "stdev_s": statistics.stdev(raw_times) if len(raw_times) > 1 else 0.0,
                "min_s": sorted_t[0],
                "max_s": sorted_t[-1],
                "p95_s": _percentile(sorted_t, 0.95),
                "p99_s": _percentile(sorted_t, 0.99),
            },
            "gflops": gflops,
        })
    return results


def bench_torch_matmul(
    sizes: list[tuple[int, int, int]],
    warmup: int = 3,
    iterations: int = 20,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Benchmark torch matrix multiplication (CPU or MPS)."""
    logger.info("bench_torch_matmul starting with sizes=%s, warmup=%s, iterations=%s, device=%s", sizes, warmup, iterations, device)
    try:
        import torch
    except ImportError:
        return []

    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return []

    results: list[dict[str, Any]] = []
    for M, K, N in sizes:
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(warmup):
            _ = A @ B
        if device == "mps":
            torch.mps.synchronize()

        raw_times: list[float] = []
        for _ in range(iterations):
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            _ = A @ B
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            raw_times.append(elapsed)

        sorted_t = sorted(raw_times)
        mean_s = statistics.mean(raw_times)
        ops = 2.0 * M * K * N
        gflops = ops / (mean_s * 1e9) if mean_s > 0 else 0.0

        results.append({
            "name": f"torch_matmul_fp16_{device}",
            "dimensions": {"M": M, "K": K, "N": N},
            "flops": ops,
            "warmup": warmup,
            "iterations": iterations,
            "raw_times_s": raw_times,
            "summary": {
                "mean_s": mean_s,
                "median_s": statistics.median(raw_times),
                "stdev_s": statistics.stdev(raw_times) if len(raw_times) > 1 else 0.0,
                "min_s": sorted_t[0],
                "max_s": sorted_t[-1],
                "p95_s": _percentile(sorted_t, 0.95),
                "p99_s": _percentile(sorted_t, 0.99),
            },
            "gflops": gflops,
        })
    return results


def bench_memory_bandwidth(
    size_mb: int = 512,
    warmup: int = 3,
    iterations: int = 15,
) -> dict[str, Any]:
    """Estimate sustained memory bandwidth via numpy array copy."""
    logger.info("bench_memory_bandwidth starting with size_mb=%s, warmup=%s, iterations=%s", size_mb, warmup, iterations)
    import numpy as np

    n_elements = (size_mb * 1024 * 1024) // 4  # float32
    arr = np.random.randn(n_elements).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = arr.copy()

    raw_times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = arr.copy()
        elapsed = time.perf_counter() - start
        raw_times.append(elapsed)

    sorted_t = sorted(raw_times)
    mean_s = statistics.mean(raw_times)
    bytes_copied = n_elements * 4 * 2  # read + write
    bandwidth_gbs = bytes_copied / (mean_s * 1e9) if mean_s > 0 else 0.0

    return {
        "name": "numpy_memory_bandwidth",
        "size_mb": size_mb,
        "warmup": warmup,
        "iterations": iterations,
        "raw_times_s": raw_times,
        "summary": {
            "mean_s": mean_s,
            "median_s": statistics.median(raw_times),
            "stdev_s": statistics.stdev(raw_times) if len(raw_times) > 1 else 0.0,
            "p95_s": _percentile(sorted_t, 0.95),
        },
        "bandwidth_gbs": bandwidth_gbs,
        "bytes_copied": bytes_copied,
    }


def bench_vector_ops(
    size: int = 10_000_000,
    warmup: int = 3,
    iterations: int = 20,
) -> dict[str, Any]:
    """Benchmark basic vector operations (add, multiply, reduce) via numpy."""
    logger.info("bench_vector_ops starting with size=%s, warmup=%s, iterations=%s", size, warmup, iterations)
    import numpy as np

    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)

    ops: dict[str, list[float]] = {}

    for op_name, fn in [
        ("add", lambda: a + b),
        ("multiply", lambda: a * b),
        ("dot", lambda: np.dot(a, b)),
        ("sum_reduce", lambda: np.sum(a)),
    ]:
        # Warmup
        for _ in range(warmup):
            _ = fn()

        raw: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = fn()
            raw.append(time.perf_counter() - start)
        ops[op_name] = raw

    ops_summary: dict[str, dict[str, float]] = {}
    for op_name, raw in ops.items():
        sorted_t = sorted(raw)
        ops_summary[op_name] = {
            "mean_ms": statistics.mean(raw) * 1000,
            "median_ms": statistics.median(raw) * 1000,
            "p95_ms": _percentile(sorted_t, 0.95) * 1000,
        }

    return {
        "name": "numpy_vector_ops",
        "size": size,
        "warmup": warmup,
        "iterations": iterations,
        "raw_times_s": ops,
        "summary": ops_summary,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("main starting")
    parser = argparse.ArgumentParser(
        description="Reproducible metal_marlin benchmark suite",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run fewer iterations for CI/quick checks",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Optional tag to identify this benchmark run",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_REPORTS_DIR,
        help="Directory to write JSON output (default: reports/)",
    )
    args = parser.parse_args()

    warmup = 2 if args.quick else 5
    iterations = 8 if args.quick else 30

    # GEMM sizes: (M, K, N)
    # - decode-like (M=1), prefill-like (M=32,128), and large (M=512)
    gemm_sizes = [
        (1, 4096, 4096),
        (1, 8192, 4096),
        (32, 4096, 4096),
        (128, 4096, 4096),
        (512, 4096, 4096),
    ]

    if args.quick:
        gemm_sizes = gemm_sizes[:3]

    print("=" * 70)
    print("Metal Marlin Reproducible Benchmark")
    print("=" * 70)
    print(f"  Mode:     {'quick' if args.quick else 'full'}")
    print(f"  Warmup:   {warmup}")
    print(f"  Iters:    {iterations}")
    print(f"  Tag:      {args.tag or '(none)'}")
    print()

    # ── Capture environment ──────────────────────────────────────────────
    print("[1/5] Capturing hardware info...")
    hw = _hardware_info()
    print(f"       Platform: {hw['platform']}")
    print(f"       Chip:     {hw.get('apple_chip', 'N/A')}")
    print(f"       Memory:   {hw.get('total_memory_gb', 'N/A')} GB")

    print("[2/5] Capturing software versions...")
    sw = _software_versions()
    print(f"       numpy:    {sw.get('numpy')}")
    print(f"       torch:    {sw.get('torch')}")

    print("[3/5] Capturing git state...")
    git = _git_info()
    if git["available"]:
        print(f"       Commit:   {git['commit'][:12]}")
        print(f"       Dirty:    {len(git['dirty_files'])} files")
    else:
        print("       (git unavailable)")

    # ── Run benchmarks ───────────────────────────────────────────────────
    print()
    print("[4/5] Running benchmarks...")

    all_results: list[dict[str, Any]] = []

    # NumPy matmul
    print("       numpy_matmul...")
    numpy_results = bench_numpy_matmul(gemm_sizes, warmup=warmup, iterations=iterations)
    all_results.extend(numpy_results)

    # Torch CPU matmul
    print("       torch_matmul (cpu)...")
    torch_cpu_results = bench_torch_matmul(
        gemm_sizes, warmup=warmup, iterations=iterations, device="cpu",
    )
    all_results.extend(torch_cpu_results)

    # Torch MPS matmul (if available)
    print("       torch_matmul (mps)...")
    torch_mps_results = bench_torch_matmul(
        gemm_sizes, warmup=warmup, iterations=iterations, device="mps",
    )
    all_results.extend(torch_mps_results)

    # Memory bandwidth
    print("       memory_bandwidth...")
    mem_bw = bench_memory_bandwidth(
        size_mb=256 if args.quick else 512,
        warmup=warmup,
        iterations=iterations,
    )
    all_results.append(mem_bw)

    # Vector ops
    print("       vector_ops...")
    vec_ops = bench_vector_ops(
        size=5_000_000 if args.quick else 10_000_000,
        warmup=warmup,
        iterations=iterations,
    )
    all_results.append(vec_ops)

    # ── Assemble report ──────────────────────────────────────────────────
    now_utc = datetime.now(UTC)
    timestamp_local = datetime.now()

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark_script": str(Path(__file__).relative_to(_PROJECT_ROOT)),
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_local": timestamp_local.isoformat(),
        "tag": args.tag,
        "mode": "quick" if args.quick else "full",
        "environment": {
            "hardware": hw,
            "software": sw,
            "git": git,
        },
        "config": {
            "warmup": warmup,
            "iterations": iterations,
            "gemm_sizes": gemm_sizes,
        },
        "results": all_results,
    }

    # Compute integrity hash (before adding the hash itself)
    integrity_hash = _compute_integrity_hash(report)
    report["integrity_hash"] = integrity_hash

    # ── Write output ─────────────────────────────────────────────────────
    print()
    print("[5/5] Writing report...")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"bench_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"       Written: {output_path}")
    print(f"       Size:    {output_path.stat().st_size:,} bytes")
    print(f"       Hash:    {integrity_hash[:16]}...")
    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)

    # Print summary table
    print()
    print(f"{'Benchmark':<35} {'Mean (ms)':>12} {'GFLOPS':>10}")
    print("-" * 59)
    for r in all_results:
        if "gflops" in r:
            dims = r.get("dimensions", {})
            dim_str = "x".join(str(dims.get(d, "")) for d in ("M", "K", "N"))
            label = f"{r['name']} [{dim_str}]"
            mean_ms = r["summary"]["mean_s"] * 1000
            gflops = r["gflops"]
            print(f"{label:<35} {mean_ms:>12.3f} {gflops:>10.2f}")
        elif r["name"] == "numpy_memory_bandwidth":
            mean_ms = r["summary"]["mean_s"] * 1000
            bw = r["bandwidth_gbs"]
            print(f"{'memory_bandwidth':<35} {mean_ms:>12.3f} {bw:>8.1f} GB/s")


if __name__ == "__main__":
    main()
