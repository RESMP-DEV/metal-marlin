#!/usr/bin/env python3
"""Roofline model analysis for Metal Marlin kernels.

This script builds a roofline plot (arithmetic intensity vs TFLOPS) and
exports JSON analysis for optimization targeting.

Example input JSON format:
{
  "kernels": [
    {"name": "gemm_fp16_4096", "type": "gemm", "M": 128, "N": 4096, "K": 4096, "tflops": 8.2},
    {
      "name": "attention_seq4096",
      "type": "attention",
      "batch": 1,
      "heads": 32,
      "seq_q": 4096,
      "seq_k": 4096,
      "head_dim": 128,
      "tflops": 5.1
    },
    {"name": "moe_dispatch", "type": "moe_dispatch", "tflops": 0.05}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_MATPLOTLIB, plt  # noqa: E402


@dataclass
class HardwareConfig:
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    memory_bw_gbs: float
    gpu_name: str

    @property
    def ridge_point(self) -> float:
        return (self.peak_fp16_tflops * 1000.0) / self.memory_bw_gbs


def _gemm_ai(
    M: int,
    N: int,
    K: int,
    *,
    bytes_per_element: float = 2.0,
) -> tuple[float, float, float]:
    flops = 2.0 * M * N * K
    bytes_total = bytes_per_element * (M * K + K * N + M * N)
    ai = flops / bytes_total if bytes_total > 0 else 0.0
    return ai, flops, bytes_total


def _attention_ai(
    *,
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    bytes_per_element: float = 2.0,
) -> tuple[float, float, float]:
    flops = 4.0 * batch * heads * seq_q * seq_k * head_dim
    bytes_total = (
        bytes_per_element
        * batch
        * heads
        * head_dim
        * (2 * seq_q + 2 * seq_k)
    )
    ai = flops / bytes_total if bytes_total > 0 else 0.0
    return ai, flops, bytes_total


def _compute_tflops(
    *,
    entry: dict[str, Any],
    flops: float,
    ai: float,
) -> float:
    if "tflops" in entry and entry["tflops"] is not None:
        return float(entry["tflops"])
    if "elapsed_ms" in entry and entry["elapsed_ms"]:
        elapsed_s = float(entry["elapsed_ms"]) / 1000.0
        return (flops / elapsed_s) / 1e12 if elapsed_s > 0 else 0.0
    if "memory_gb_s" in entry and entry["memory_gb_s"]:
        return (ai * float(entry["memory_gb_s"])) / 1000.0
    return 0.0


def _analyze_point(
    *,
    name: str,
    ai: float,
    tflops: float,
    config: HardwareConfig,
) -> dict[str, Any]:
    ridge = config.ridge_point
    if ai < ridge:
        bound = "memory"
        max_tflops = (ai * config.memory_bw_gbs) / 1000.0
    else:
        bound = "compute"
        max_tflops = config.peak_fp16_tflops
    attainment = (tflops / max_tflops) * 100 if max_tflops > 0 else 0.0
    return {
        "name": name,
        "bound": bound,
        "tflops": tflops,
        "arithmetic_intensity": ai,
        "max_achievable_tflops": max_tflops,
        "attainment_pct": attainment,
        "headroom_tflops": max_tflops - tflops,
        "distance_to_ridge": abs(ai - ridge),
    }


def _load_kernels_from_results(results_dir: Path) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                if {"M", "N", "K", "tflops"}.issubset(item.keys()):
                    kernels.append(
                        {
                            "name": item.get("name", path.stem),
                            "type": "gemm",
                            "M": int(item["M"]),
                            "N": int(item["N"]),
                            "K": int(item["K"]),
                            "tflops": float(item["tflops"]),
                        }
                    )
    return kernels


def _parse_inputs(input_path: Path | None, results_dir: Path) -> list[dict[str, Any]]:
    if input_path is not None:
        with open(input_path) as f:
            payload = json.load(f)
        kernels = payload.get("kernels", []) if isinstance(payload, dict) else payload
        if not isinstance(kernels, list):
            raise ValueError("Input JSON must contain a list of kernels")
        return [k for k in kernels if isinstance(k, dict)]

    kernels = _load_kernels_from_results(results_dir)
    if kernels:
        return kernels

    raise RuntimeError(
        "No kernel inputs found. Provide --input with kernel definitions or run "
        "benchmarks that export JSON to benchmarks/results/*.json."
    )


def _plot_roofline(
    *,
    kernels: list[dict[str, Any]],
    config: HardwareConfig,
    output_path: Path,
) -> None:
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    ai_values = [k["arithmetic_intensity"] for k in kernels]
    min_ai = max(min(ai_values) * 0.5, 1e-2)
    max_ai = max(ai_values) * 2.0

    ai_range = (min_ai, max_ai)
    ai = np.logspace(np.log10(ai_range[0]), np.log10(ai_range[1]), 500)
    memory_roof = (ai * config.memory_bw_gbs) / 1000.0
    compute_roof = np.full_like(ai, config.peak_fp16_tflops)
    fp32_roof = np.full_like(ai, config.peak_fp32_tflops)

    fig, ax = plt.subplots(figsize=(12, 8))
    roof = np.minimum(memory_roof, compute_roof)
    ax.loglog(ai, roof, "b-", linewidth=2.5, label=f"Roofline ({config.gpu_name})")

    memory_region = ai < config.ridge_point
    ax.loglog(
        ai[memory_region],
        memory_roof[memory_region],
        "b--",
        linewidth=1,
        alpha=0.5,
    )
    ax.loglog(
        ai[~memory_region],
        compute_roof[~memory_region],
        "b--",
        linewidth=1,
        alpha=0.5,
    )

    ax.loglog(ai, np.minimum(memory_roof, fp32_roof), "g--", linewidth=1, label="FP32")

    ax.axvline(
        config.ridge_point,
        color="gray",
        linestyle=":",
        alpha=0.7,
        linewidth=1,
    )
    ax.annotate(
        f"Ridge: {config.ridge_point:.1f} F/B",
        xy=(config.ridge_point, config.peak_fp16_tflops * 0.3),
        fontsize=9,
        color="gray",
    )

    colors = plt.cm.Set1.colors  # type: ignore[attr-defined]
    for idx, kernel in enumerate(kernels):
        ax.plot(
            kernel["arithmetic_intensity"],
            kernel["tflops"],
            "o",
            color=colors[idx % len(colors)],
            markersize=9,
            label=kernel["label"],
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(f"Roofline Model - {config.gpu_name}", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(ai_range)
    ax.set_ylim(0.1, config.peak_fp16_tflops * 1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roofline analysis for Metal Marlin kernels")
    parser.add_argument("--input", type=Path, help="Optional JSON file with kernel definitions")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Output directory for plot and JSON",
    )
    parser.add_argument("--peak-fp16", type=float, default=14.0, help="Peak FP16 TFLOPS")
    parser.add_argument("--peak-fp32", type=float, default=7.0, help="Peak FP32 TFLOPS")
    parser.add_argument("--memory-bw", type=float, default=400.0, help="Memory BW GB/s")
    parser.add_argument("--gpu-name", type=str, default="M4 Max (example)")
    args = parser.parse_args()

    config = HardwareConfig(
        peak_fp16_tflops=args.peak_fp16,
        peak_fp32_tflops=args.peak_fp32,
        memory_bw_gbs=args.memory_bw,
        gpu_name=args.gpu_name,
    )

    kernels_in = _parse_inputs(args.input, args.output_dir)

    kernels: list[dict[str, Any]] = []
    for entry in kernels_in:
        name = str(entry.get("name", "kernel"))
        kernel_type = str(entry.get("type", "generic")).lower()
        label = str(entry.get("label", name))
        bytes_per_element = float(entry.get("bytes_per_element", 2.0))

        ai = 0.0
        flops = float(entry.get("flops", 0.0))
        bytes_moved = float(entry.get("bytes_moved", 0.0))

        if kernel_type == "gemm":
            ai, flops, bytes_moved = _gemm_ai(
                int(entry["M"]),
                int(entry["N"]),
                int(entry["K"]),
                bytes_per_element=bytes_per_element,
            )
        elif kernel_type == "attention":
            ai, flops, bytes_moved = _attention_ai(
                batch=int(entry.get("batch", 1)),
                heads=int(entry.get("heads", 1)),
                seq_q=int(entry["seq_q"]),
                seq_k=int(entry["seq_k"]),
                head_dim=int(entry["head_dim"]),
                bytes_per_element=bytes_per_element,
            )
        elif "arithmetic_intensity" in entry:
            ai = float(entry["arithmetic_intensity"])

        if ai == 0.0 and flops > 0 and bytes_moved > 0:
            ai = flops / bytes_moved

        tflops = _compute_tflops(entry=entry, flops=flops, ai=ai)

        kernels.append(
            {
                "name": name,
                "label": label,
                "type": kernel_type,
                "tflops": tflops,
                "arithmetic_intensity": ai,
                "flops": flops,
                "bytes_moved": bytes_moved,
                "metadata": {
                    k: v
                    for k, v in entry.items()
                    if k
                    not in {
                        "name",
                        "label",
                        "type",
                        "tflops",
                        "arithmetic_intensity",
                        "flops",
                        "bytes_moved",
                    }
                },
            }
        )

    analyses = [_analyze_point(name=k["name"], ai=k["arithmetic_intensity"], tflops=k["tflops"], config=config) for k in kernels]

    optimization_targets = {
        "below_roofline": [
            a["name"] for a in analyses if a["attainment_pct"] < 50.0
        ],
        "memory_bound": [a["name"] for a in analyses if a["bound"] == "memory"],
        "compute_bound": [a["name"] for a in analyses if a["bound"] == "compute"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "roofline_analysis.png"
    json_path = args.output_dir / "roofline_analysis.json"

    _plot_roofline(kernels=kernels, config=config, output_path=plot_path)

    output = {
        "config": {
            "gpu_name": config.gpu_name,
            "peak_fp16_tflops": config.peak_fp16_tflops,
            "peak_fp32_tflops": config.peak_fp32_tflops,
            "memory_bw_gbs": config.memory_bw_gbs,
            "ridge_point": config.ridge_point,
        },
        "kernels": [
            {
                **k,
                "analysis": a,
            }
            for k, a in zip(kernels, analyses, strict=True)
        ],
        "optimization_targets": optimization_targets,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved roofline plot to {plot_path}")
    print(f"Saved roofline analysis to {json_path}")


if __name__ == "__main__":
    main()
