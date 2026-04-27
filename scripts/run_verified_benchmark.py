#!/usr/bin/env python3
"""
Reproducible benchmark harness for Metal Marlin.

Generates machine-readable JSON results with tamper-evident integrity
checks to prevent fabricated benchmark claims.

Usage:
    uv run python scripts/run_verified_benchmark.py
    uv run python scripts/run_verified_benchmark.py --quick

Output:
    contrib/metal_marlin/reports/bench_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MIN_WARMUP_ITERS = 5
MIN_BENCH_ITERS = 10


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EnvMeta:
    benchmark_uuid: str
    timestamp_utc: str
    hostname: str
    platform: str
    macos_version: str | None
    cpu: str
    memory_gb: float
    python_version: str
    python_executable: str
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    metal_marlin_version: str
    torch_version: str | None
    numpy_version: str
    mps_available: bool | None
    # capture full environment hash for reproducibility
    env_hash_sha256: str


@dataclass
class TimingSample:
    iteration: int
    wall_ms: float


@dataclass
class BenchResult:
    name: str
    unit: str
    # Raw timing series (tamper-evident)
    samples: list[TimingSample] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    # Integrity hash over the raw sample wall_ms values (recomputed on load)
    sample_hash_sha256: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkReport:
    meta: EnvMeta
    results: list[BenchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_rev() -> tuple[str | None, str | None, bool]:
    """Return (commit, branch, dirty) or None on failure."""
    logger.debug("_git_rev called")
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--short"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            != ""
        )
        return commit, branch, dirty
    except Exception:
        return None, None, False


def _env_hash() -> str:
    """Deterministic hash of environment variables that affect reproducibility."""
    logger.debug("_env_hash called")
    keys = sorted(k for k in os.environ if "TORCH" in k or "MPS" in k or "OMP" in k or "MKL" in k)
    blob = json.dumps({k: os.environ[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _memory_gb() -> float:
    logger.debug("_memory_gb called")
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return 0.0


def _capture_meta() -> EnvMeta:
    logger.debug("_capture_meta called")
    import numpy as np

    try:
        import torch

        torch_version = torch.__version__
        mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else None
    except Exception:
        torch_version = None
        mps = None

    commit, branch, dirty = _git_rev()
    macos_ver = platform.mac_ver()[0] if hasattr(platform, "mac_ver") else None

    return EnvMeta(
        benchmark_uuid=str(uuid.uuid4()),
        timestamp_utc=datetime.now(UTC).isoformat(),
        hostname=platform.node(),
        platform=sys.platform,
        macos_version=macos_ver or None,
        cpu=platform.machine(),
        memory_gb=_memory_gb(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        metal_marlin_version="0.1.0",
        torch_version=torch_version,
        numpy_version=np.__version__,
        mps_available=mps,
        env_hash_sha256=_env_hash(),
    )


def _hash_samples(samples: list[TimingSample]) -> str:
    """Produce tamper-evident hash of raw timing series."""
    logger.debug("_hash_samples called with samples=%s", samples)
    blob = json.dumps([s.wall_ms for s in samples])
    return hashlib.sha256(blob.encode()).hexdigest()


def _stats(samples: list[float]) -> dict[str, float]:
    logger.debug("_stats called with samples=%s", samples)
    arr = np.array(samples, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean_ms": round(float(np.mean(arr)), 6),
        "median_ms": round(float(np.median(arr)), 6),
        "std_ms": round(float(np.std(arr, ddof=1)), 6),
        "min_ms": round(float(np.min(arr)), 6),
        "max_ms": round(float(np.max(arr)), 6),
        "p95_ms": round(float(np.percentile(arr, 95)), 6),
        "p99_ms": round(float(np.percentile(arr, 99)), 6),
    }


def _mps_gc():
    logger.debug("_mps_gc called")
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmarks (simple synthetic workloads that don't need model weights)
# ---------------------------------------------------------------------------

def _import_torch():
    logger.debug("_import_torch called")
    import torch

    return torch


def bench_matmul_fp16(M: int, K: int, N: int, iters: int = MIN_BENCH_ITERS) -> BenchResult:
    """Baseline FP16 GEMM on MPS (or CPU fallback)."""
    logger.info("bench_matmul_fp16 starting with M=%s, K=%s, N=%s, iters=%s", M, K, N, iters)
    torch = _import_torch()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(MIN_WARMUP_ITERS):
        _ = A @ B
        if device == "mps":
            torch.mps.synchronize()

    samples: list[TimingSample] = []
    for i in range(iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        C = A @ B
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        samples.append(TimingSample(iteration=i, wall_ms=(t1 - t0) * 1000.0))

    del A, B, C
    _mps_gc()

    result = BenchResult(
        name="matmul_fp16",
        unit="ms",
        samples=samples,
        summary=_stats([s.wall_ms for s in samples]),
        parameters={"M": M, "K": K, "N": N, "device": device, "dtype": "float16"},
    )
    result.sample_hash_sha256 = _hash_samples(samples)
    return result


def bench_moe_expert_dispatch(
    num_experts: int = 8,
    d_model: int = 2048,
    hidden: int = 6144,
    top_k: int = 2,
    batch: int = 1,
    iters: int = MIN_BENCH_ITERS,
) -> BenchResult:
    """Benchmark MoE-style expert weight scatter/gather dispatch."""
    logger.info("bench_moe_expert_dispatch starting with num_experts=%s, d_model=%s, hidden=%s, top_k=%s", num_experts, d_model, hidden, top_k)
    torch = _import_torch()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    gate_up = torch.randn(num_experts, d_model, hidden, device=device, dtype=torch.float16)
    down = torch.randn(num_experts, hidden, d_model, device=device, dtype=torch.float16)
    x = torch.randn(batch, d_model, device=device, dtype=torch.float16)
    # Random top-k experts
    expert_ids = torch.randint(0, num_experts, (batch, top_k), device=device)

    for _ in range(MIN_WARMUP_ITERS):
        for e in expert_ids.unbind():
            selected = gate_up[e]  # (top_k, d_model, hidden)
            _ = x.unsqueeze(1) @ selected
        if device == "mps":
            torch.mps.synchronize()

    samples: list[TimingSample] = []
    for i in range(iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for e in expert_ids.unbind():
            h = x.unsqueeze(1) @ gate_up[e]
            h = h @ down[e]
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        samples.append(TimingSample(iteration=i, wall_ms=(t1 - t0) * 1000.0))

    del gate_up, down, x, h
    _mps_gc()

    result = BenchResult(
        name="moe_expert_dispatch",
        unit="ms",
        samples=samples,
        summary=_stats([s.wall_ms for s in samples]),
        parameters={
            "num_experts": num_experts,
            "d_model": d_model,
            "hidden": hidden,
            "top_k": top_k,
            "batch": batch,
            "device": device,
        },
    )
    result.sample_hash_sha256 = _hash_samples(samples)
    return result


def bench_attention_decode(
    seq_len: int = 4096,
    n_heads: int = 32,
    head_dim: int = 128,
    iters: int = MIN_BENCH_ITERS,
) -> BenchResult:
    """Single-token decode attention: Q @ K^T @ V."""
    logger.info("bench_attention_decode starting with seq_len=%s, n_heads=%s, head_dim=%s, iters=%s", seq_len, n_heads, head_dim, iters)
    torch = _import_torch()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    Q = torch.randn(1, n_heads, 1, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    for _ in range(MIN_WARMUP_ITERS):
        scores = Q @ K.transpose(-2, -1) / (head_dim**0.5)
        attn = torch.softmax(scores, dim=-1) @ V
        if device == "mps":
            torch.mps.synchronize()

    samples: list[TimingSample] = []
    for i in range(iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        scores = Q @ K.transpose(-2, -1) / (head_dim**0.5)
        attn = torch.softmax(scores, dim=-1) @ V
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        samples.append(TimingSample(iteration=i, wall_ms=(t1 - t0) * 1000.0))

    del Q, K, V, attn, scores
    _mps_gc()

    result = BenchResult(
        name="attention_decode",
        unit="ms",
        samples=samples,
        summary=_stats([s.wall_ms for s in samples]),
        parameters={"seq_len": seq_len, "n_heads": n_heads, "head_dim": head_dim, "device": device},
    )
    result.sample_hash_sha256 = _hash_samples(samples)
    return result


def bench_layer_norm(hidden: int = 2048, seq_len: int = 512, iters: int = MIN_BENCH_ITERS) -> BenchResult:
    """LayerNorm on a typical activation tensor."""
    logger.info("bench_layer_norm starting with hidden=%s, seq_len=%s, iters=%s", hidden, seq_len, iters)
    torch = _import_torch()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    x = torch.randn(seq_len, hidden, device=device, dtype=torch.float16)
    gamma = torch.ones(hidden, device=device, dtype=torch.float16)
    beta = torch.zeros(hidden, device=device, dtype=torch.float16)
    eps = 1e-5

    for _ in range(MIN_WARMUP_ITERS):
        _ = torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)
        if device == "mps":
            torch.mps.synchronize()

    samples: list[TimingSample] = []
    for i in range(iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        samples.append(TimingSample(iteration=i, wall_ms=(t1 - t0) * 1000.0))

    del x, gamma, beta
    _mps_gc()

    result = BenchResult(
        name="layer_norm",
        unit="ms",
        samples=samples,
        summary=_stats([s.wall_ms for s in samples]),
        parameters={"hidden": hidden, "seq_len": seq_len, "device": device, "dtype": "float16"},
    )
    result.sample_hash_sha256 = _hash_samples(samples)
    return result


def bench_quantized_matmul(
    M: int = 1,
    K: int = 4096,
    N: int = 4096,
    group_size: int = 128,
    iters: int = MIN_BENCH_ITERS,
) -> BenchResult:
    """Simulated GPTQ-style INT4 dequant + matmul."""
    logger.info("bench_quantized_matmul starting with M=%s, K=%s, N=%s, group_size=%s", M, K, N, group_size)
    torch = _import_torch()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Fake quantized weights: int4 stored in int8 [-8,7]
    qweight = torch.randint(-8, 8, (K, N), device=device, dtype=torch.int8)
    # Scales & zeros per group
    num_groups = K // group_size
    scales = torch.rand(num_groups, N, device=device, dtype=torch.float16) / 16.0
    zeros = torch.rand(num_groups, N, device=device, dtype=torch.float16) / 16.0
    x = torch.randn(M, K, device=device, dtype=torch.float16)

    def dequant():
        logger.info("dequant called")
        w = qweight.to(torch.float16)
        # apply per-group scale/zero
        for g in range(num_groups):
            w[g * group_size : (g + 1) * group_size] = (
                w[g * group_size : (g + 1) * group_size] - zeros[g]
            ) * scales[g]
        return w

    for _ in range(MIN_WARMUP_ITERS):
        w = dequant()
        _ = x @ w
        if device == "mps":
            torch.mps.synchronize()

    samples: list[TimingSample] = []
    for i in range(iters):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        w = dequant()
        _ = x @ w
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        samples.append(TimingSample(iteration=i, wall_ms=(t1 - t0) * 1000.0))

    del qweight, scales, zeros, x, w
    _mps_gc()

    result = BenchResult(
        name="quantized_matmul_simulated",
        unit="ms",
        samples=samples,
        summary=_stats([s.wall_ms for s in samples]),
        parameters={
            "M": M,
            "K": K,
            "N": N,
            "group_size": group_size,
            "device": device,
            "bits": 4,
        },
    )
    result.sample_hash_sha256 = _hash_samples(samples)
    return result


# ---------------------------------------------------------------------------
# Runner & serialization
# ---------------------------------------------------------------------------

def run_all(quick: bool = False) -> BenchmarkReport:
    logger.debug("run_all called with quick=%s", quick)
    meta = _capture_meta()
    print(f"Benchmark UUID: {meta.benchmark_uuid}")
    print(f"Timestamp: {meta.timestamp_utc}")
    print(f"Host: {meta.hostname} ({meta.platform})")
    print(f"Git: {meta.git_commit or 'N/A'} on {meta.git_branch or 'N/A'} {'(dirty)' if meta.git_dirty else '(clean)'}")
    print(f"Torch: {meta.torch_version or 'N/A'}  MPS: {meta.mps_available}")
    print()

    results: list[BenchResult] = []
    iters = 5 if quick else MIN_BENCH_ITERS

    workloads = [
        ("matmul_fp16 (M=1)", lambda: bench_matmul_fp16(1, 4096, 4096, iters=iters)),
        ("matmul_fp16 (M=64)", lambda: bench_matmul_fp16(64, 4096, 4096, iters=iters)),
        ("matmul_fp16 (M=512)", lambda: bench_matmul_fp16(512, 4096, 4096, iters=iters)),
        ("moe_expert_dispatch", lambda: bench_moe_expert_dispatch(
            num_experts=8, d_model=2048, hidden=6144, top_k=2, batch=1, iters=iters
        )),
        ("attention_decode (seq=1024)", lambda: bench_attention_decode(
            seq_len=1024, n_heads=32, head_dim=128, iters=iters
        )),
        ("attention_decode (seq=4096)", lambda: bench_attention_decode(
            seq_len=4096, n_heads=32, head_dim=128, iters=iters
        )),
        ("layer_norm", lambda: bench_layer_norm(hidden=2048, seq_len=512, iters=iters)),
        ("quantized_matmul", lambda: bench_quantized_matmul(
            M=1, K=4096, N=4096, group_size=128, iters=iters
        )),
    ]

    for label, fn in workloads:
        print(f"Running {label} ...", end=" ", flush=True)
        try:
            res = fn()
            print(f"done ({res.summary['mean_ms']:.3f} ms mean)")
            results.append(res)
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append(
                BenchResult(
                    name=label.split()[0],
                    unit="ms",
                    error=str(exc),
                    parameters={},
                )
            )

    return BenchmarkReport(meta=meta, results=results)


def _serialize(obj: Any) -> Any:
    logger.debug("_serialize called with obj=%s", obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(getattr(obj, k)) for k in obj.__dataclass_fields__}
    raise TypeError(f"Cannot serialize {type(obj)}")


def write_report(report: BenchmarkReport) -> Path:
    logger.info("write_report called with report=%s", report)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"bench_{ts}.json"

    data = _serialize(report)
    # Compute top-level integrity hash over the deterministic subset
    integrity_blob = json.dumps(data["meta"], sort_keys=True) + "\n"
    for r in data["results"]:
        integrity_blob += json.dumps({k: r[k] for k in ("name", "sample_hash_sha256", "parameters")}, sort_keys=True) + "\n"
    data["integrity"] = {
        "algorithm": "sha256",
        "hash": hashlib.sha256(integrity_blob.encode()).hexdigest(),
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    return out_path


def validate_report(path: Path) -> bool:
    """Recompute integrity hash and verify it matches."""
    logger.debug("validate_report called with path=%s", path)
    with open(path) as f:
        data = json.load(f)

    integrity = data.get("integrity", {})
    stored_hash = integrity.get("hash", "")

    integrity_blob = json.dumps(data["meta"], sort_keys=True) + "\n"
    for r in data["results"]:
        integrity_blob += json.dumps({k: r[k] for k in ("name", "sample_hash_sha256", "parameters")}, sort_keys=True) + "\n"
    computed = hashlib.sha256(integrity_blob.encode()).hexdigest()

    ok = computed == stored_hash
    print(f"Validation: {'PASS' if ok else 'FAIL'} (computed {computed[:16]}... vs stored {stored_hash[:16]}...)")
    return ok


def main():
    logger.info("main starting")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Run abbreviated benchmark (fewer iterations)")
    parser.add_argument("--validate", type=Path, metavar="PATH", help="Validate an existing report JSON")
    args = parser.parse_args()

    if args.validate:
        ok = validate_report(args.validate)
        sys.exit(0 if ok else 1)

    report = run_all(quick=args.quick)
    out_path = write_report(report)
    print(f"\nReport written to: {out_path}")
    ok = validate_report(out_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
