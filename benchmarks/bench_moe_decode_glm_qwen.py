#!/usr/bin/env python3
"""Decode regression benchmark for GLM-4.7 MoE and Qwen3 Coder Next MoE.

This benchmark reports per-preset decode metrics:
- decode_ms_per_token
- decode_tok_per_s
- fallback/grouping diagnostics when exposed by runtime

Default output file:
    contrib/metal_marlin/benchmarks/results/moe_decode_glm_qwen_regression.json
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

# Standalone contrib project import path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_TORCH, torch  # noqa: E402
from metal_marlin.trellis.lm import TrellisForCausalLM  # noqa: E402

if not HAS_TORCH or torch is None:
    raise RuntimeError("PyTorch is required to run this benchmark.")

SUPPORTED_PRESETS = ("glm47_moe", "qwen3_coder_next")
DEFAULT_PRESETS = list(SUPPORTED_PRESETS)

PRESET_CANDIDATES: dict[str, list[str]] = {
    "glm47_moe": [
        "models/GLM-4.7-Flash-Marlin-MMFP4",
        "models/GLM-4.7-Flash-Marlin-MMFP4",
        "models/GLM-4.7-Flash-Trellis",
    ],
    "qwen3_coder_next": [
        "models/Qwen3-Coder-Next-Trellis-3bpw",
        "models/Qwen3-Coder-Next-Trellis",
    ],
}


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


def _make_fallback_collectors() -> tuple[list[Callable[[], None]], dict[str, Callable[[], Any]], list[str]]:
    resetters: list[Callable[[], None]] = []
    collectors: dict[str, Callable[[], Any]] = {}
    unavailable: list[str] = []

    try:
        from metal_marlin.trellis.moe_dispatch import (  # noqa: WPS433
            get_mixed_bpw_grouping_fallback_diagnostics,
            reset_mixed_bpw_grouping_fallback_counters,
        )

        resetters.append(reset_mixed_bpw_grouping_fallback_counters)
        collectors["mixed_bpw_grouping"] = get_mixed_bpw_grouping_fallback_diagnostics
    except Exception as exc:  # pragma: no cover - optional diagnostics
        unavailable.append(f"mixed_bpw_grouping:{exc.__class__.__name__}")

    try:
        from metal_marlin.trellis.metrics import moe_metrics  # noqa: WPS433

        if hasattr(moe_metrics, "reset"):
            resetters.append(moe_metrics.reset)

        def _collect_moe_metrics() -> dict[str, int]:
            result: dict[str, int] = {}
            fields = ("fallback_used", "fast_path_used", "tokens_processed", "nan_detected")
            for field in fields:
                obj = getattr(moe_metrics, field, None)
                value = getattr(obj, "value", None)
                if isinstance(value, (int, float)):
                    result[field] = int(value)
            return result

        collectors["moe_metrics"] = _collect_moe_metrics
    except Exception as exc:  # pragma: no cover - optional diagnostics
        unavailable.append(f"moe_metrics:{exc.__class__.__name__}")

    return resetters, collectors, unavailable


def _collect_diagnostics(collectors: dict[str, Callable[[], Any]]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for name, fn in collectors.items():
        try:
            value = fn()
        except Exception:
            continue
        if value:
            result[name] = value
    return result or None


def _parse_presets(args: argparse.Namespace) -> list[str]:
    presets: list[str] = []
    for raw in args.presets.split(","):
        value = raw.strip()
        if value:
            presets.append(value)
    for value in args.preset:
        value = value.strip()
        if value:
            presets.append(value)

    if not presets:
        presets = list(DEFAULT_PRESETS)

    deduped: list[str] = []
    for preset in presets:
        if preset not in deduped:
            deduped.append(preset)

    invalid = [preset for preset in deduped if preset not in SUPPORTED_PRESETS]
    if invalid:
        supported = ", ".join(SUPPORTED_PRESETS)
        requested = ", ".join(invalid)
        raise ValueError(f"Unsupported preset(s): {requested}. Supported presets: {supported}.")

    return deduped


def _parse_model_path_mappings(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --model-path '{raw}'. Use format '<preset>=<path>'.")
        preset, path = raw.split("=", 1)
        preset = preset.strip()
        path = path.strip()
        if not preset or not path:
            raise ValueError(f"Invalid --model-path '{raw}'. Use format '<preset>=<path>'.")
        if preset not in SUPPORTED_PRESETS:
            supported = ", ".join(SUPPORTED_PRESETS)
            raise ValueError(
                f"Unsupported preset in --model-path '{preset}'. Supported presets: {supported}."
            )
        overrides[preset] = path
    return overrides


def _find_model_path(preset: str, overrides: dict[str, str]) -> tuple[str, str]:
    override = overrides.get(preset)
    if override:
        return override, "cli-override"

    candidates = PRESET_CANDIDATES.get(preset, [])
    for rel_path in candidates:
        abs_path = _ROOT / rel_path
        if abs_path.exists() and abs_path.is_dir():
            return str(abs_path), "default:first-existing-candidate"

    if candidates:
        return str(_ROOT / candidates[0]), "default:first-candidate"
    return "", "not-found"


def run_benchmark_for_preset(
    preset: str,
    model_path: str,
    device: str,
    prompt_len: int,
    decode_tokens: int,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    if not model_path or not Path(model_path).exists():
        return {
            "status": "error",
            "reason": f"Model path does not exist: {model_path}",
            "preset": preset,
            "model_path": model_path,
        }

    if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
        return {
            "status": "skipped",
            "reason": "ALPHAHENG_TASK_MODE=1",
            "preset": preset,
            "model_path": model_path,
        }

    torch.manual_seed(seed)

    try:
        load_start = time.perf_counter()
        model = TrellisForCausalLM.from_pretrained(model_path, device=device)
        model.eval()
        _sync_device(device)
        model_load_s = time.perf_counter() - load_start

        vocab_size = int(getattr(model.config, "vocab_size", 32000))
        vocab_size = max(vocab_size, 2)
        prompt_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)
        decode_input = prompt_ids[:, -1:].contiguous()

        resetters, collectors, unavailable_diagnostics = _make_fallback_collectors()

        with torch.inference_mode():
            _ = model(prompt_ids)  # prefill
            _sync_device(device)

            for _ in range(warmup):
                _ = model(decode_input)
            _sync_device(device)

            for reset_fn in resetters:
                try:
                    reset_fn()
                except Exception:
                    continue

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

        total_tokens = decode_tokens * runs
        total_time_s = sum(per_run_s)
        ms_per_token = (total_time_s * 1000.0) / max(total_tokens, 1)
        tok_per_s = total_tokens / max(total_time_s, 1e-9)

        return {
            "status": "ok",
            "preset": preset,
            "model_path": model_path,
            "model_load_s": model_load_s,
            "decode_ms_per_token": ms_per_token,
            "decode_tok_per_s": tok_per_s,
            "p50_ms": _percentile(per_token_ms, 0.50),
            "p95_ms": _percentile(per_token_ms, 0.95),
            "p99_ms": _percentile(per_token_ms, 0.99),
            "std_ms": statistics.pstdev(per_token_ms) if len(per_token_ms) > 1 else 0.0,
            "fallback_diagnostics": _collect_diagnostics(collectors),
            "diagnostic_sources_unavailable": unavailable_diagnostics,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "preset": preset,
            "model_path": model_path,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE decode regression benchmark (GLM/Qwen).")
    parser.add_argument(
        "--presets",
        default=",".join(DEFAULT_PRESETS),
        help="Comma-separated presets to run (default: glm47_moe,qwen3_coder_next).",
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Additional single preset to run; can be repeated.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Model path override in format '<preset>=<path>'; can be repeated.",
    )
    parser.add_argument("--glm-model-path", help="Override model path for preset 'glm47_moe'.")
    parser.add_argument(
        "--qwen-model-path",
        help="Override model path for preset 'qwen3_coder_next'.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto|mps|cuda|cpu.")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path "
            "(default: contrib/metal_marlin/benchmarks/results/moe_decode_glm_qwen_regression.json)"
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.prompt_len < 1 or args.decode_tokens < 1 or args.warmup < 0 or args.runs < 1:
        parser.error(
            "Invalid benchmark settings: require prompt_len>=1, decode_tokens>=1, warmup>=0, runs>=1."
        )

    presets = _parse_presets(args)
    device = _resolve_device(args.device)

    overrides = _parse_model_path_mappings(args.model_path)
    if args.glm_model_path:
        overrides["glm47_moe"] = args.glm_model_path
    if args.qwen_model_path:
        overrides["qwen3_coder_next"] = args.qwen_model_path

    results: dict[str, dict[str, Any]] = {}
    for preset in presets:
        model_path, source = _find_model_path(preset, overrides)
        print(f"--- Running benchmark for preset: {preset} ---")
        print(f"Model path: {model_path} ({source})")

        result = run_benchmark_for_preset(
            preset=preset,
            model_path=model_path,
            device=device,
            prompt_len=args.prompt_len,
            decode_tokens=args.decode_tokens,
            warmup=args.warmup,
            runs=args.runs,
            seed=args.seed,
        )
        result["model_path_source"] = source
        results[preset] = result

        if result["status"] == "ok":
            print(
                "  decode_ms_per_token="
                f"{result['decode_ms_per_token']:.3f}, decode_tok_per_s={result['decode_tok_per_s']:.3f}"
            )
        else:
            print(f"  status={result['status']} reason={result.get('reason')}")

    summary: dict[str, Any] = {
        "benchmark": "moe_decode_glm_qwen_regression",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_presets": presets,
        "settings": {
            "device_requested": args.device,
            "device_used": device,
            "prompt_len": args.prompt_len,
            "decode_tokens": args.decode_tokens,
            "warmup": args.warmup,
            "runs": args.runs,
            "seed": args.seed,
            "torch_version": torch.__version__,
        },
        "results": results,
    }

    output_path = args.output
    if not output_path:
        output_path = str(_ROOT / "benchmarks" / "results" / "moe_decode_glm_qwen_regression.json")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
