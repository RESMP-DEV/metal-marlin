#!/usr/bin/env python3
"""End-to-end inference script for Qwen3.6-35B-A3B with MMFP4 quantized weights on M4 (MPS).

Loads a pre-quantized checkpoint, runs a simple generation benchmark,
and prints latency / throughput / memory metrics with a hard sanity gate
on decode speed.

All architecture-specific defaults and the inference pipeline come from
``qwen_delta_inference_common``.  This wrapper only selects the Qwen3.6
config and delegates to the shared entry point.

Usage:
    cd contrib/metal_marlin
    uv run python scripts/run_qwen36_inference.py

Environment:
    QWEN36_MODEL_PATH   Override default checkpoint path.
    HF_TOKEN            HuggingFace token (only needed if tokenizer must be downloaded).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – allow importing from the scripts/ and metal_marlin package.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from qwen_delta_inference_common import (  # noqa: E402
    QWEN36_35B_A3B_CONFIG,
    build_inference_argparser,
    run_inference_main,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expose defaults for external introspection / testing
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / QWEN36_35B_A3B_CONFIG.default_model_path
DEFAULT_TOKENIZER = QWEN36_35B_A3B_CONFIG.default_tokenizer
DEFAULT_PROMPT = "Once upon a time"
DEFAULT_MAX_NEW_TOKENS = 128
MIN_DECODE_SPEED_TOK_S = QWEN36_35B_A3B_CONFIG.min_decode_speed

# Re-export parser so ``_parse_args`` is still available for tests
_parse_args = build_inference_argparser(QWEN36_35B_A3B_CONFIG, "QWEN36").parse_args


def main(argv: list[str] | None = None) -> int:
    logger.info("main starting")
    return run_inference_main(argv, QWEN36_35B_A3B_CONFIG, "QWEN36")


if __name__ == "__main__":
    raise SystemExit(main())
