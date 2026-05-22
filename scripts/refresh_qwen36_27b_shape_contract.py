#!/usr/bin/env python3
"""Refresh Qwen3.6-27B shape-contract evidence from Hugging Face config."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metal_marlin.qwen36_27b_profile import (  # noqa: E402
    CONFIG_URL,
    MODEL_ID,
    profile_from_hf_config,
    shape_contract_payload,
    validate_supported_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Qwen/Qwen3.6-27B config metadata and write shape_contract.json."
    )
    parser.add_argument("--config-url", default=CONFIG_URL)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "agent_workspace" / "qwen36_27b" / "shape_contract.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with urlopen(args.config_url, timeout=30) as response:
        config = json.loads(response.read().decode("utf-8"))

    profile = profile_from_hf_config(config)
    validate_supported_profile(profile)
    payload = shape_contract_payload(profile)
    payload["config_fetched_at"] = datetime.now(UTC).date().isoformat()
    payload["weights_loaded"] = False
    payload["weights_loaded_note"] = (
        "No safetensors shards were downloaded; only config metadata was fetched."
    )
    payload["model_id"] = MODEL_ID

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

