#!/usr/bin/env python3
"""Reproducible benchmark for Metal Marlin decode latency.

Generates machine-readable JSON with integrity hashes so that every result can
be independently verified, preventing fabricated benchmark claims.

Benchmark specification:
- Fixed input: "The quick brown fox jumps over the lazy dog."
- 3 warmup decode steps (discarded timing)
- N decode steps with per-step timing
- Statistics: mean, median, std, min, max, p5, p95, p99 of per-step latency
- Corresponding tok/s computed for each percentile

Usage:
    uv run python contrib/metal_marlin/bench_reproducible.py \
        --model-path ./models/glm47-flash-mmfp4 \
        --num-steps 100 \
        --output-json bench_results.json

Arguments:
    --model-path      Path to a quantised model directory (required)
    --num-steps       Number of measured decode steps (default: 100)
    --output-json     Path for the JSON results file (required)
"""

from __future__ import annotations

import argparse
import gc
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
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root so that ``metal_marlin`` is importable regardless of cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecodeStepTiming:
    """Per decode step timing."""
    step: int
    latency_ms: float           # wall-clock for this decode step
    tokens_per_sec: float       # 1000 / latency_ms


@dataclass
class EnvironmentInfo:
    """Full environment snapshot for reproducibility."""
    benchmark_uuid: str
    timestamp_utc: str
    hostname: str
    platform: str
    macos_version: str | None
    python_version: str
    python_executable: str
    torch_version: str | None
    mps_available: bool | None
    apple_chip: str | None
    physical_cpus: int | None
    logical_cpus: int | None
    total_memory_gb: float
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    env_hash: str


@dataclass
class TimingRun:
    """Per-iteration timing breakdown."""
    iteration: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    ttft_ms: float
    tokens_generated: int
    tps_decode: float
    tps_overall: float
    per_step_timings_ms: list[float] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Model information for reproducibility."""
    model_path: str
    quantization_type: str | None
    parameter_count: int | None
    model_sha256: str | None


@dataclass
class BenchmarkReport:
    """Complete, tamper-evident benchmark report."""
    schema_version: str = "3.0"
    script: str = "bench_reproducible.py"
    script_sha256: str | None = None
    environment: EnvironmentInfo | None = None
    model: ModelInfo | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    runs: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _git_info() -> tuple[str | None, str | None, bool]:
    logger.debug("_git_info called")
    try:
        def _run(cmd: list[str]) -> str:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(_PROJECT_ROOT), timeout=10)
            return r.stdout.strip()
        commit = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]).strip())
        return commit or None, branch or None, dirty
    except Exception:
        return None, None, False


def _env_hash() -> str:
    logger.debug("_env_hash called")
    keys = sorted(k for k in os.environ
                  if any(x in k for x in
                         ("TORCH", "MPS", "OMP", "MKL", "METAL", "PYTORCH")))
    blob = json.dumps({k: os.environ[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _script_sha256() -> str:
    """Compute SHA256 of this script file for tamper detection."""
    logger.debug("_script_sha256 called")
    script_path = Path(__file__).resolve()
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _get_model_info(model_path: str) -> ModelInfo:
    """Extract model information including quantization type and parameter count."""
    logger.debug("_get_model_info called with model_path=%s", model_path)
    model_dir = Path(model_path)
    quant_type = None
    param_count = None
    model_hash = None

    # Try to detect quantization type from config or file patterns
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            # Check for quantization indicators
            if "quantization_config" in config:
                qc = config["quantization_config"]
                quant_type = qc.get("quant_method", qc.get("qmethod", "unknown"))
            elif "awq_version" in config:
                quant_type = "awq"
            elif "gptq" in str(config).lower():
                quant_type = "gptq"
        except Exception:
            pass

    # Check for MMFP4 quantization files
    if any(model_dir.glob("*.mmfp4")) or any(model_dir.glob("*-mmfp4*")):
        quant_type = "mmfp4"

    # Count parameters by summing weight file sizes (approximate)
    # or read from config if available
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if "num_parameters" in config:
                param_count = config["num_parameters"]
            elif "hidden_size" in config and "num_hidden_layers" in config:
                # Rough estimate based on transformer architecture
                h = config.get("hidden_size", 0)
                l = config.get("num_hidden_layers", 0)
                a = config.get("intermediate_size", h * 4)
                # Approximate: layers * (hidden^2 * 2 + intermediate * hidden)
                param_count = l * (h * h * 2 + a * h)
        except Exception:
            pass

    # Compute SHA256 of model index or first weight file
    try:
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "rb") as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()[:32]
        else:
            # Hash first safetensors file
            for st_file in sorted(model_dir.glob("*.safetensors")):
                with open(st_file, "rb") as f:
                    # Just hash first 1MB for speed
                    model_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()[:32]
                break
    except Exception:
        pass

    return ModelInfo(
        model_path=model_path,
        quantization_type=quant_type,
        parameter_count=param_count,
        model_sha256=model_hash,
    )


def capture_environment() -> EnvironmentInfo:
    """Collect full hardware/software environment metadata."""
    logger.debug("capture_environment called")
    hw: dict[str, Any] = {}
    if sys.platform == "darwin":
        try:
            def _sysctl(key: str) -> str:
                logger.debug("_sysctl called with key=%s", key)
                r = subprocess.run(["sysctl", "-n", key],
                                   capture_output=True, text=True, timeout=5)
                return r.stdout.strip()
            hw["apple_chip"] = _sysctl("machdep.cpu.brand_string")
            hw["physical_cpus"] = int(_sysctl("hw.physicalcpu"))
            hw["logical_cpus"] = int(_sysctl("hw.logicalcpu"))
            hw["total_memory_gb"] = round(int(_sysctl("hw.memsize")) / (1024**3), 2)
            hw["macos_version"] = platform.mac_ver()[0]
        except Exception:
            hw["apple_chip"] = None
            hw["physical_cpus"] = None
            hw["logical_cpus"] = None
            hw["total_memory_gb"] = 0.0
            hw["macos_version"] = platform.mac_ver()[0] if sys.platform == "darwin" else None

    torch_version: str | None = None
    mps_available: bool | None = None
    try:
        import torch  # noqa: F811
        torch_version = torch.__version__
        mps_available = bool(getattr(torch.backends, "mps", None)
                             and torch.backends.mps.is_available())
    except ImportError:
        pass

    commit, branch, dirty = _git_info()

    return EnvironmentInfo(
        benchmark_uuid=str(uuid.uuid4()),
        timestamp_utc=datetime.now(UTC).isoformat(),
        hostname=platform.node(),
        platform=platform.platform(),
        macos_version=hw.get("macos_version"),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        torch_version=torch_version,
        mps_available=mps_available,
        apple_chip=hw.get("apple_chip"),
        physical_cpus=hw.get("physical_cpus"),
        logical_cpus=hw.get("logical_cpus"),
        total_memory_gb=hw.get("total_memory_gb", 0.0),
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        env_hash=_env_hash(),
    )


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _device() -> str:
    logger.debug("_device called")
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _sync(device: str) -> None:
    logger.debug("_sync called with device=%s", device)
    try:
        import torch
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _clear(device: str) -> None:
    logger.debug("_clear called with device=%s", device)
    gc.collect()
    try:
        import torch
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model loading — tries multiple backends gracefully
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str):
    """Return ``(model_or_pipeline, tokenizer, generate_fn)``.

    *generate_fn* has the signature::

        generate_fn(model, input_ids, max_new_tokens) -> output_ids
    """
    logger.info("load_model called with model_path=%s, device=%s", model_path, device)
    errors: list[str] = []

    # 1. MMFP4Pipeline (from metal_marlin.inference.mmfp4_pipeline)
    try:
        from transformers import AutoTokenizer

        from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline
        pipeline = MMFP4Pipeline.from_pretrained(model_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        def _gen(m, ids, n):
            return m.generate(ids, max_new_tokens=n, do_sample=False)
        return pipeline, tokenizer, _gen
    except Exception as exc:
        errors.append(f"MMFP4Pipeline: {exc}")

    # 2. TrellisForCausalLM (from metal_marlin.trellis)
    try:
        from transformers import AutoTokenizer

        from metal_marlin.trellis import TrellisForCausalLM
        model = TrellisForCausalLM.from_pretrained(model_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        def _gen(m, ids, n):
            return m.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
        return model, tokenizer, _gen
    except Exception as exc:
        errors.append(f"TrellisForCausalLM: {exc}")

    # 3. MetalInferenceEngine (from metal_marlin.inference)
    try:
        from metal_marlin.inference import MetalInferenceEngine
        engine = MetalInferenceEngine(model_path, device=device)
        tokenizer = engine.tokenizer
        def _gen(m, ids, n):
            import torch
            with torch.no_grad():
                out = m.model.generate(ids, max_new_tokens=n, do_sample=False)
            return out
        return engine, tokenizer, _gen
    except Exception as exc:
        errors.append(f"MetalInferenceEngine: {exc}")

    raise RuntimeError(
        "Could not load model via any known backend.\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Core benchmark — per-step decode timing
# ---------------------------------------------------------------------------

# Fixed prompt as specified in task requirements
BENCHMARK_PROMPT = "The quick brown fox jumps over the lazy dog."

WARMUP_STEPS = 3  # fixed warmup decode steps (discarded)


def run_decode_benchmark(
    model,
    tokenizer,
    generate_fn,
    device: str,
    prompt: str,
    num_steps: int,
    warmup_steps: int = WARMUP_STEPS,
) -> list[DecodeStepTiming]:
    """Execute decode step benchmark with per-step timing.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        generate_fn: Generate function with signature (model, input_ids, max_new_tokens)
        device: Device string ("mps", "cuda", or "cpu")
        prompt: Input prompt string
        num_steps: Number of measured decode steps
        warmup_steps: Number of warmup decode steps (discarded), default=3

    Returns:
        List of DecodeStepTiming for each measured step
    """
    logger.info("run_decode_benchmark starting with model=%s, tokenizer=%s, generate_fn=%s, device=%s", model, tokenizer, generate_fn, device)
    import torch

    # Tokenize prompt once
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup: generate warmup_steps tokens (discard timing)
    if warmup_steps > 0:
        _ = generate_fn(model, input_ids, warmup_steps)
        _sync(device)
        _clear(device)

    # Start fresh from prompt for measured steps
    current_ids = input_ids.clone()

    # Measure each decode step individually
    timings: list[DecodeStepTiming] = []

    for step in range(num_steps):
        _sync(device)
        step_start = time.perf_counter()

        # Generate exactly one new token
        with torch.no_grad():
            output = generate_fn(model, current_ids, 1)

        _sync(device)
        step_end = time.perf_counter()

        latency_ms = (step_end - step_start) * 1000.0
        tps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        timings.append(DecodeStepTiming(
            step=step,
            latency_ms=round(latency_ms, 4),
            tokens_per_sec=round(tps, 2),
        ))

        # Update current_ids for next step
        current_ids = output

    return timings


def _percentile(sorted_vals: list[float], pct: float) -> float:
    logger.debug("_percentile called with sorted_vals=%s, pct=%s", sorted_vals, pct)
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summarize(values: list[float]) -> dict[str, float]:
    """Compute statistics including mean, median, std, min, max, p5, p95, p99."""
    logger.debug("_summarize called with values=%s", values)
    if not values:
        return {}
    s = sorted(values)
    return {
        "n": len(s),
        "mean": round(statistics.mean(s), 4),
        "median": round(statistics.median(s), 4),
        "std": round(statistics.stdev(s), 4) if len(s) > 1 else 0.0,
        "min": round(min(s), 4),
        "max": round(max(s), 4),
        "p5": round(_percentile(s, 0.05), 4),
        "p95": round(_percentile(s, 0.95), 4),
        "p99": round(_percentile(s, 0.99), 4),
    }


def compute_integrity(report_dict: dict) -> str:
    """SHA-256 over environment + configuration + raw run hashes."""
    logger.debug("compute_integrity called with report_dict=%s", report_dict)
    parts: list[str] = []
    if report_dict.get("environment"):
        parts.append(json.dumps(report_dict["environment"], sort_keys=True, default=str))
    if report_dict.get("configuration"):
        parts.append(json.dumps(report_dict["configuration"], sort_keys=True, default=str))
    # Hash per-step latency_ms values to detect tampering with raw data.
    for run in report_dict.get("runs", []):
        parts.append(f"step:{run['step']}:{run['latency_ms']}")
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Run iteration
# ---------------------------------------------------------------------------

def run_iteration(
    model,
    tokenizer,
    generate_fn,
    device: str,
    prompt: str,
    num_tokens: int,
    warmup_tokens: int,
    iteration: int,
) -> TimingRun:
    """Run a single benchmark iteration with prefill + decode timing.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        generate_fn: Generate function with signature (model, input_ids, max_new_tokens)
        device: Device string ("mps", "cuda", or "cpu")
        prompt: Input prompt string
        num_tokens: Number of tokens to generate (measured)
        warmup_tokens: Number of warmup tokens (discarded)
        iteration: Iteration index

    Returns:
        TimingRun with timing breakdown
    """
    logger.debug("run_iteration called with model=%s, tokenizer=%s, generate_fn=%s", model, tokenizer, generate_fn)
    import torch

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Warmup (discarded)
    if warmup_tokens > 0:
        _ = generate_fn(model, input_ids, warmup_tokens)
        _sync(device)
        _clear(device)

    # Measure prefill time (first token after prompt)
    _sync(device)
    prefill_start = time.perf_counter()
    with torch.no_grad():
        first_token_output = generate_fn(model, input_ids, 1)
    _sync(device)
    prefill_end = time.perf_counter()
    prefill_ms = (prefill_end - prefill_start) * 1000.0

    # TTFT is essentially the prefill time for first token
    ttft_ms = prefill_ms

    # Measure decode time for remaining tokens
    current_ids = first_token_output
    decode_start = time.perf_counter()

    for _ in range(num_tokens - 1):
        with torch.no_grad():
            current_ids = generate_fn(model, current_ids, 1)

    _sync(device)
    decode_end = time.perf_counter()
    decode_ms = (decode_end - decode_start) * 1000.0

    # Total time
    total_ms = prefill_ms + decode_ms

    # Calculate tokens per second
    tokens_generated = num_tokens
    tps_decode = (num_tokens - 1) / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    tps_overall = tokens_generated / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return TimingRun(
        iteration=iteration,
        prefill_ms=round(prefill_ms, 4),
        decode_ms=round(decode_ms, 4),
        total_ms=round(total_ms, 4),
        ttft_ms=round(ttft_ms, 4),
        tokens_generated=tokens_generated,
        tps_decode=round(tps_decode, 2),
        tps_overall=round(tps_overall, 2),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_NUM_STEPS = 100  # default number of measured decode steps


def main() -> None:
    logger.info("main starting")
    parser = argparse.ArgumentParser(
        description="Reproducible Metal Marlin decode benchmark with per-step timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark specification:
  - Fixed prompt: "The quick brown fox jumps over the lazy dog."
  - Warmup: 3 decode steps (timing discarded)
  - Measured: N decode steps with per-step timing
  - Statistics: mean, median, std, min, max, p5, p95, p99
  - Output: Machine-readable JSON with integrity hash
        """,
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the quantised model directory")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"Number of measured decode steps (default: {DEFAULT_NUM_STEPS})")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Output path for the JSON results file")
    args = parser.parse_args()

    device = _device()

    # ── header ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("METAL MARLIN — REPRODUCIBLE DECODE BENCHMARK")
    print("=" * 64)
    print(f"  Model:        {args.model_path}")
    print(f"  Device:       {device}")
    print(f"  Warmup steps: {WARMUP_STEPS} (discarded)")
    print(f"  Measured steps: {args.num_steps}")
    print(f"  Output:       {args.output_json}")
    print("=" * 64)

    # ── capture environment ─────────────────────────────────────────────
    print("\n[1/3] Capturing environment …")
    env = capture_environment()
    print(f"  UUID:      {env.benchmark_uuid}")
    print(f"  Platform:  {env.platform}")
    print(f"  Torch:     {env.torch_version or 'N/A'}")
    print(f"  MPS:       {env.mps_available}")
    if env.apple_chip:
        print(f"  Chip:      {env.apple_chip}")
    print(f"  Memory:    {env.total_memory_gb} GB")
    print(f"  Git:       {(env.git_commit or 'N/A')[:12]}"
          f"{' (dirty)' if env.git_dirty else ' (clean)'}")

    # ── load model ──────────────────────────────────────────────────────
    print("\n[2/3] Loading model …")
    try:
        model, tokenizer, generate_fn = load_model(args.model_path, device)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  FAILED: {exc}")
        model_info = _get_model_info(args.model_path)
        report = BenchmarkReport(
            environment=env,
            model=model_info,
            configuration={
                "model_path": args.model_path,
                "num_steps": args.num_steps,
                "warmup_steps": WARMUP_STEPS,
                "device": device,
                "prompt": BENCHMARK_PROMPT,
                "error_loading": True,
            },
        )
        report_dict = asdict(report)
        report_dict["script_sha256"] = _script_sha256()
        report_dict["error"] = {"message": str(exc), "traceback": tb}
        report_dict["integrity"] = {
            "algorithm": "sha256",
            "hash": compute_integrity(report_dict),
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
            f.write("\n")
        print(f"\nError report saved to: {out_path}")
        sys.exit(1)

    prompt_tokens = tokenizer(BENCHMARK_PROMPT, return_tensors="pt").input_ids.shape[1]
    print(f"  Loaded. Prompt tokens: {prompt_tokens}")

    # ── run decode benchmark ────────────────────────────────────────────
    print(f"\n[3/3] Running {args.num_steps} decode steps …")
    print(f"  (First {WARMUP_STEPS} warmup steps discarded)")

    timings = run_decode_benchmark(
        model=model,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        device=device,
        prompt=BENCHMARK_PROMPT,
        num_steps=args.num_steps,
        warmup_steps=WARMUP_STEPS,
    )

    # ── aggregate statistics ────────────────────────────────────────────
    latency_values = [t.latency_ms for t in timings]
    tps_values = [t.tokens_per_sec for t in timings]

    latency_stats = _summarize(latency_values)
    tps_stats = _summarize(tps_values)

    # Compute tok/s for each percentile latency
    latency_percentiles = {
        "mean": latency_stats.get("mean", 0),
        "median": latency_stats.get("median", 0),
        "p5": latency_stats.get("p5", 0),
        "p95": latency_stats.get("p95", 0),
        "p99": latency_stats.get("p99", 0),
    }
    tps_from_latency = {
        k: round(1000.0 / v, 2) if v > 0 else 0.0
        for k, v in latency_percentiles.items()
    }

    configuration = {
        "model_path": args.model_path,
        "num_steps": args.num_steps,
        "warmup_steps": WARMUP_STEPS,
        "device": device,
        "prompt": BENCHMARK_PROMPT,
    }

    # Get model info
    model_info = _get_model_info(args.model_path)

    # Build report
    report = BenchmarkReport(
        environment=env,
        model=model_info,
        configuration=configuration,
        runs=[{"step": t.step, "latency_ms": t.latency_ms, "tokens_per_sec": t.tokens_per_sec} for t in timings],
        summary={
            "latency_ms": latency_stats,
            "tokens_per_sec": tps_stats,
            "toks_per_sec_from_latency": tps_from_latency,
            "prompt_tokens": prompt_tokens,
        },
    )

    # ── integrity hash ──────────────────────────────────────────────────
    report_dict = asdict(report)
    report_dict["script_sha256"] = _script_sha256()
    report_dict["integrity"] = {
        "algorithm": "sha256",
        "hash": compute_integrity(report_dict),
        "description": (
            "SHA-256 over environment + configuration + raw step timings. "
            "Any post-hoc editing of those fields invalidates this hash."
        ),
    }

    # ── write JSON ──────────────────────────────────────────────────────
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
        f.write("\n")

    # ── human-readable summary ──────────────────────────────────────────
    print()
    print("=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  Decode steps measured: {args.num_steps}")
    print()
    print("  Latency (ms/step):")
    print(f"    Mean:   {latency_stats.get('mean', 0):.2f} ± {latency_stats.get('std', 0):.2f} ms")
    print(f"    Median: {latency_stats.get('median', 0):.2f} ms")
    print(f"    Min:    {latency_stats.get('min', 0):.2f} ms")
    print(f"    Max:    {latency_stats.get('max', 0):.2f} ms")
    print(f"    p5:     {latency_stats.get('p5', 0):.2f} ms")
    print(f"    p95:    {latency_stats.get('p95', 0):.2f} ms")
    print(f"    p99:    {latency_stats.get('p99', 0):.2f} ms")
    print()
    print("  Throughput (tok/s):")
    print(f"    Mean:   {tps_stats.get('mean', 0):.2f} ± {tps_stats.get('std', 0):.2f} tok/s")
    print(f"    Median: {tps_stats.get('median', 0):.2f} tok/s")
    print()
    print("  Tok/s from latency percentiles:")
    for k, v in tps_from_latency.items():
        print(f"    {k}: {v:.2f} tok/s")
    print()
    print(f"  Integrity: {report_dict['integrity']['hash'][:24]}…")
    print(f"  Saved to:  {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
