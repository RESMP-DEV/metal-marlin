#!/usr/bin/env python3
"""Launch the Qwen3.6-35B-A3B OpenAI-compatible server.

This wrapper uses the shared Qwen DeltaNet serving helper with Qwen3.6-35B-A3B
defaults and wires serving to ``MMFP4Pipeline`` (which uses the Trellis model
for hybrid DeltaNet/attention + MoE architecture).

Qwen3.6-35B-A3B Architecture:
    - 256 routed experts, top-8 active (~3.5B active params)
    - Hybrid attention: DeltaNet linear attention + full GQA with layer-wise
      scheduling via ``full_attention_interval`` list
    - MLA (Multi-Head Latent Attention) with compressed KV
    - Vocab size: 248,320 (compatible with Qwen3.5 tokenizer)

Examples:
    QWEN36_MODEL_PATH=models/Qwen3.6-35B-A3B-MMFP4 \\
    uv run python scripts/serve_qwen36.py

    uv run python scripts/serve_qwen36.py \\
      --model-path models/Qwen3.6-35B-A3B-MMFP4 \\
      --port 8000 --batch-size 32

Test:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello"}]}'
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – allow importing from the scripts/ and metal_marlin package.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from qwen_delta_serving_common import (  # noqa: E402
    QwenDeltaServeConfig,
    add_serving_args,
    build_config,
    install_mmfp4_adapter,
    run_serving_server,
    validate_serving_args,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3.6-35B-A3B defaults
# ---------------------------------------------------------------------------

_ENV_PREFIX = "QWEN36"
"""Upper-case prefix for environment-variable defaults."""
_DEFAULT_MODEL_PATH = "models/Qwen3.6-35B-A3B-MMFP4"
"""Default checkpoint path for Qwen3.6-35B-A3B MMFP4."""

# Expose for external introspection / testing
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / _DEFAULT_MODEL_PATH


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    logger.info("_build_parser starting")
    parser = argparse.ArgumentParser(
        description="Launch Qwen3.6-35B-A3B OpenAI server with MMFP4 + PagedAttention defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_serving_args(parser, _ENV_PREFIX, default_model_path=_DEFAULT_MODEL_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    logger.info("main starting")
    parser = _build_parser()
    args = parser.parse_args(argv)
    validate_serving_args(args, _ENV_PREFIX)
    config: QwenDeltaServeConfig = build_config(args, _ENV_PREFIX)

    install_mmfp4_adapter(
        enable_persistent_cache=True,
        use_paged_attention=config.use_paged_attention,
    )

    return run_serving_server(config)


if __name__ == "__main__":
    raise SystemExit(main())
