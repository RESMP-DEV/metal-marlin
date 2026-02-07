#!/usr/bin/env python3
"""Inspect availability of MoE Metal kernel symbols and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metal_marlin.metal_dispatch import MetalKernelLibrary

KERNEL_SYMBOLS = [
    "moe_trellis_swiglu_decode",
    "moe_trellis_swiglu_prefill4",
    "moe_trellis_swiglu_prefill4_fp32acc",
    "moe_trellis_swiglu",
    "moe_trellis_swiglu_fp32acc",
    "moe_trellis_swiglu_large_batch",
    "moe_trellis_swiglu_grouped",
    "moe_trellis_swiglu_decode_6_2_3",
    "moe_trellis_swiglu_decode_6_3_4",
    "moe_trellis_swiglu_decode_6_2_4",
]

GLM47_DOMINANT_TUPLE_KERNEL = "moe_trellis_swiglu_decode_6_2_3"

DEFAULT_OUTPUT_PATH = (
    ROOT / "benchmarks" / "results" / "moe_kernel_symbol_inventory.json"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect MoE Metal kernel symbol availability."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write JSON output (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def inspect_kernel_symbols(lib: MetalKernelLibrary) -> dict[str, Any]:
    """Check whether each target kernel symbol resolves to a Metal pipeline."""
    available_kernels: list[str] = []
    missing_kernels: list[str] = []

    for symbol in KERNEL_SYMBOLS:
        try:
            lib.get_pipeline(symbol)
        except Exception:
            missing_kernels.append(symbol)
        else:
            available_kernels.append(symbol)

    return {
        "available_kernels": available_kernels,
        "missing_kernels": missing_kernels,
        "glm47_dominant_tuple_kernel_available": (
            GLM47_DOMINANT_TUPLE_KERNEL in available_kernels
        ),
    }


def main() -> int:
    """Run kernel availability inspection and write JSON output."""
    args = parse_args()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    lib = MetalKernelLibrary.from_source_dir()
    report = inspect_kernel_symbols(lib)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote kernel inventory to: {output_path}")
    print(
        "available="
        f"{len(report['available_kernels'])} "
        "missing="
        f"{len(report['missing_kernels'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
