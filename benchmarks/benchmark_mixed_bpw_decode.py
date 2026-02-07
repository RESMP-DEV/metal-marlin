#!/usr/bin/env python3
"""Focused benchmark for mixed-BPW decode throughput and fallback counters.

Emits a JSON summary with:
- Decode latency (ms/token)
- Decode throughput (tok/s)
- Mixed-BPW metadata (if config is available)
- Fallback counters (if runtime exposes them)

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/benchmark_mixed_bpw_decode.py \
        --model-path models/GLM-4.7-Flash-Trellis-3bpw

Model/config can be provided via CLI or env vars:
    MIXED_BPW_MODEL_PATH
    MODEL_PATH
    METAL_MARLIN_MODEL_PATH
    MIXED_BPW_QUANT_CONFIG
    METAL_MARLIN_QUANT_CONFIG
    MIXED_BPW_BENCH_CONFIG   (optional JSON benchmark config file)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Standalone contrib project import path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_TORCH, torch  # noqa: E402
from metal_marlin.trellis.lm import TrellisForCausalLM  # noqa: E402

if not HAS_TORCH or torch is None:
    raise RuntimeError("PyTorch is required to run this benchmark.")


_PRECISION_BITS: dict[str, float] = {
    "fp16": 16.0,
    "bf16": 16.0,
    "fp8": 8.0,
    "fp8e4m3": 8.0,
    "fp4": 4.0,
    "fp4e2m1": 4.0,
    "int8": 8.0,
    "int4": 4.0,
    "int3": 3.0,
    "int2": 2.0,
    "nf3": 3.0,
    "nf2": 2.0,
}


def _normalize_precision_key(key: str) -> str:
    return key.strip().lower().replace("-", "").replace("_", "")


def _precision_to_bits(key: str) -> float | None:
    normalized = _normalize_precision_key(key)
    if normalized in _PRECISION_BITS:
        return _PRECISION_BITS[normalized]
    if normalized.isdigit():
        return float(int(normalized))
    if normalized.endswith("bit") and normalized[:-3].isdigit():
        return float(int(normalized[:-3]))
    return None


def _sync_device(device: str) -> None:
    if device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((len(sorted_vals) - 1) * q)
    return float(sorted_vals[max(0, min(idx, len(sorted_vals) - 1))])


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _pick_str(
    cli_value: str | None,
    env_names: list[str],
    cfg: dict[str, Any],
    cfg_key: str,
    default: str | None = None,
) -> str | None:
    if cli_value:
        return cli_value
    for env_name in env_names:
        val = os.environ.get(env_name)
        if val:
            return val
    cfg_val = cfg.get(cfg_key)
    if isinstance(cfg_val, str) and cfg_val:
        return cfg_val
    return default


def _pick_int(
    cli_value: int | None,
    env_names: list[str],
    cfg: dict[str, Any],
    cfg_key: str,
    default: int,
) -> int:
    if cli_value is not None:
        return cli_value
    for env_name in env_names:
        val = os.environ.get(env_name)
        if val is not None:
            return _safe_int(val, default)
    if cfg_key in cfg:
        return _safe_int(cfg[cfg_key], default)
    return default


def _resolve_device(requested: str) -> str:
    req = requested.strip().lower()
    if req == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req.startswith("mps") and not torch.backends.mps.is_available():
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _summarize_mixed_bpw(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None

    out: dict[str, Any] = {}

    avg_bits = config.get("average_bits")
    if isinstance(avg_bits, (int, float)):
        out["average_bpw"] = float(avg_bits)

    by_precision = config.get("by_precision")
    if isinstance(by_precision, dict):
        weighted_bits = 0.0
        total_params = 0.0
        normalized_counts: dict[str, int] = {}

        for key, count_raw in by_precision.items():
            count = _safe_int(count_raw, 0)
            if count <= 0:
                continue
            normalized_counts[str(key)] = count
            bits = _precision_to_bits(str(key))
            if bits is not None:
                weighted_bits += bits * count
                total_params += count

        if normalized_counts:
            out["by_precision"] = normalized_counts
        if total_params > 0 and "average_bpw" not in out:
            out["average_bpw"] = weighted_bits / total_params

    tensors = config.get("tensors")
    if isinstance(tensors, list) and "average_bpw" not in out:
        total_params = 0
        weighted_bits = 0.0
        for entry in tensors:
            if not isinstance(entry, dict):
                continue
            bits = entry.get("bits")
            shape = entry.get("shape")
            if not isinstance(bits, (int, float)) or not isinstance(shape, list):
                continue
            params = 1
            valid = True
            for dim in shape:
                if not isinstance(dim, int) or dim < 0:
                    valid = False
                    break
                params *= dim
            if not valid or params <= 0:
                continue
            total_params += params
            weighted_bits += float(bits) * params
        if total_params > 0:
            out["average_bpw"] = weighted_bits / total_params

    if not out:
        return None
    return out


def _make_fallback_collectors() -> tuple[list[Callable[[], None]], dict[str, Callable[[], Any]], list[str]]:
    resetters: list[Callable[[], None]] = []
    collectors: dict[str, Callable[[], Any]] = {}
    unavailable: list[str] = []

    try:
        from metal_marlin.trellis.moe import (  # noqa: WPS433
            get_moe_dispatch_stats,
            reset_moe_dispatch_stats,
        )

        resetters.append(reset_moe_dispatch_stats)
        collectors["trellis_moe_dispatch"] = get_moe_dispatch_stats
    except Exception as exc:
        unavailable.append(f"trellis_moe_dispatch:{exc.__class__.__name__}")

    try:
        from metal_marlin.trellis.metrics import moe_metrics  # noqa: WPS433

        if hasattr(moe_metrics, "reset"):
            resetters.append(moe_metrics.reset)

        def _collect_moe_metrics() -> dict[str, int]:
            result: dict[str, int] = {}
            fields = (
                "fallback_used",
                "fast_path_used",
                "tokens_processed",
                "nan_detected",
            )
            for field in fields:
                obj = getattr(moe_metrics, field, None)
                value = getattr(obj, "value", None)
                if isinstance(value, (int, float)):
                    result[field] = int(value)
            return result

        collectors["trellis_moe_metrics"] = _collect_moe_metrics
    except Exception as exc:
        unavailable.append(f"trellis_moe_metrics:{exc.__class__.__name__}")

    try:
        from metal_marlin.trellis.nan_guard import get_nan_statistics  # noqa: WPS433

        nan_stats = get_nan_statistics()
        if hasattr(nan_stats, "clear"):
            resetters.append(nan_stats.clear)
        if hasattr(nan_stats, "get_summary"):
            collectors["trellis_nan_guard"] = nan_stats.get_summary
    except Exception as exc:
        unavailable.append(f"trellis_nan_guard:{exc.__class__.__name__}")

    try:
        from metal_marlin.trellis.moe_dispatch import (  # noqa: WPS433
            get_mixed_bpw_grouping_fallback_diagnostics,
            reset_mixed_bpw_grouping_fallback_counters,
        )

        resetters.append(reset_mixed_bpw_grouping_fallback_counters)
        collectors[
            "trellis_mixed_bpw_grouping"
        ] = get_mixed_bpw_grouping_fallback_diagnostics
    except Exception as exc:
        unavailable.append(f"trellis_mixed_bpw_grouping:{exc.__class__.__name__}")

    return resetters, collectors, unavailable


def _collect_fallback_counters(collectors: dict[str, Callable[[], Any]]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for name, fn in collectors.items():
        try:
            value = fn()
        except Exception:
            continue
        if value:
            result[name] = value
    return result or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mixed-BPW decode throughput with fallback counters."
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=None,
        help="Path to quantized model directory.",
    )
    parser.add_argument(
        "--quant-config",
        "--config-path",
        dest="quant_config",
        default=None,
        help="Path to quantization JSON config (default: <model>/quantization_config.json).",
    )
    parser.add_argument(
        "--config",
        dest="bench_config",
        default=None,
        help="Optional benchmark config JSON file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on: auto|mps|cuda|cpu (default: auto).",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help="Prompt length for prefill warm start (default: 128).",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=None,
        help="Decode tokens per run (default: 128).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup decode steps (default: 8).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of timed runs (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: 1337).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to also write JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
        summary = {
            "benchmark": "mixed_bpw_decode",
            "status": "skipped",
            "reason": "ALPHAHENG_TASK_MODE=1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    bench_cfg: dict[str, Any] = {}
    bench_config_path_str = _pick_str(
        args.bench_config,
        ["MIXED_BPW_BENCH_CONFIG", "METAL_MARLIN_BENCH_CONFIG"],
        {},
        "bench_config",
    )
    if bench_config_path_str:
        loaded = _load_json(Path(bench_config_path_str))
        if loaded is not None:
            bench_cfg = loaded

    default_model = str(_ROOT / "models" / "GLM-4.7-Flash-Trellis-3bpw")
    model_path = _pick_str(
        args.model_path,
        ["MIXED_BPW_MODEL_PATH", "METAL_MARLIN_MODEL_PATH", "MODEL_PATH"],
        bench_cfg,
        "model_path",
        default=default_model,
    )
    assert model_path is not None

    quant_config_str = _pick_str(
        args.quant_config,
        ["MIXED_BPW_QUANT_CONFIG", "METAL_MARLIN_QUANT_CONFIG"],
        bench_cfg,
        "quant_config",
        default=None,
    )

    requested_device = _pick_str(
        args.device,
        ["MIXED_BPW_DEVICE", "METAL_MARLIN_DEVICE"],
        bench_cfg,
        "device",
        default="auto",
    )
    assert requested_device is not None
    device = _resolve_device(requested_device)

    prompt_len = _pick_int(
        args.prompt_len,
        ["MIXED_BPW_PROMPT_LEN", "METAL_MARLIN_PROMPT_LEN"],
        bench_cfg,
        "prompt_len",
        default=128,
    )
    decode_tokens = _pick_int(
        args.decode_tokens,
        ["MIXED_BPW_DECODE_TOKENS", "METAL_MARLIN_DECODE_TOKENS"],
        bench_cfg,
        "decode_tokens",
        default=128,
    )
    warmup = _pick_int(
        args.warmup,
        ["MIXED_BPW_WARMUP", "METAL_MARLIN_WARMUP"],
        bench_cfg,
        "warmup",
        default=8,
    )
    runs = _pick_int(
        args.runs,
        ["MIXED_BPW_RUNS", "METAL_MARLIN_BENCH_RUNS"],
        bench_cfg,
        "runs",
        default=3,
    )
    seed = _pick_int(
        args.seed,
        ["MIXED_BPW_SEED", "METAL_MARLIN_BENCH_SEED"],
        bench_cfg,
        "seed",
        default=1337,
    )

    model_dir = Path(model_path).expanduser()
    if quant_config_str:
        quant_config_path = Path(quant_config_str).expanduser()
    else:
        default_quant = model_dir / "quantization_config.json"
        quant_config_path = default_quant if default_quant.exists() else None

    quant_config = _load_json(quant_config_path) if quant_config_path else None
    mixed_bpw_summary = _summarize_mixed_bpw(quant_config)

    if prompt_len < 1 or decode_tokens < 1 or warmup < 0 or runs < 1:
        raise ValueError(
            f"Invalid benchmark args: prompt_len={prompt_len}, decode_tokens={decode_tokens}, "
            f"warmup={warmup}, runs={runs}"
        )

    torch.manual_seed(seed)

    load_start = time.perf_counter()
    model = TrellisForCausalLM.from_pretrained(str(model_dir), device=device)
    model.eval()
    _sync_device(device)
    model_load_s = time.perf_counter() - load_start

    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    vocab_size = max(vocab_size, 2)
    prompt_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    decode_input = prompt_ids[:, -1:].contiguous()

    # Prepare optional fallback-counter collectors
    resetters, collectors, unavailable_sources = _make_fallback_collectors()

    with torch.inference_mode():
        # Prefill once to initialize caches and kernels
        _ = model(prompt_ids)
        _sync_device(device)

        # Warmup decode steps
        for _ in range(warmup):
            _ = model(decode_input)
        _sync_device(device)

        # Reset counters right before timed region
        for reset_fn in resetters:
            try:
                reset_fn()
            except Exception:
                pass

        per_token_ms: list[float] = []
        per_run_s: list[float] = []

        for _ in range(runs):
            _sync_device(device)
            run_start = time.perf_counter()
            for _ in range(decode_tokens):
                _sync_device(device)
                step_start = time.perf_counter()
                _ = model(decode_input)
                _sync_device(device)
                per_token_ms.append((time.perf_counter() - step_start) * 1000.0)
            per_run_s.append(time.perf_counter() - run_start)

    total_decode_tokens = decode_tokens * runs
    total_time_s = sum(per_run_s)
    ms_per_token = (total_time_s * 1000.0) / max(total_decode_tokens, 1)
    tok_per_s = total_decode_tokens / max(total_time_s, 1e-9)

    fallback_counters = _collect_fallback_counters(collectors)

    summary: dict[str, Any] = {
        "benchmark": "mixed_bpw_decode",
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_dir),
            "quant_config_path": str(quant_config_path) if quant_config_path else None,
            "mixed_bpw": mixed_bpw_summary,
        },
        "runtime": {
            "device_requested": requested_device,
            "device_used": device,
            "torch_version": torch.__version__,
            "model_load_s": model_load_s,
        },
        "decode": {
            "prompt_len": prompt_len,
            "warmup_tokens": warmup,
            "decode_tokens_per_run": decode_tokens,
            "runs": runs,
            "total_decode_tokens": total_decode_tokens,
            "total_time_s": total_time_s,
            "ms_per_token": ms_per_token,
            "tok_per_s": tok_per_s,
            "p50_ms": _percentile(per_token_ms, 0.50),
            "p95_ms": _percentile(per_token_ms, 0.95),
            "p99_ms": _percentile(per_token_ms, 0.99),
            "std_ms": statistics.pstdev(per_token_ms) if len(per_token_ms) > 1 else 0.0,
        },
        "fallback_counters": fallback_counters,
        "fallback_counter_sources_unavailable": unavailable_sources,
    }

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
