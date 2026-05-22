#!/usr/bin/env python3
"""Import prototype qwenmetal Qwen3.6-27B qmi4 artifacts into Metal Marlin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metal_marlin.qwen36_27b_qmi4 import expected_import_output_dir, import_qmi4_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert qwenmetal .qmi4 files into split qweight/scales/zeros files "
            "plus a Metal Marlin Qwen3.6-27B artifact manifest."
        )
    )
    parser.add_argument("source_dir", help="Directory containing qwenmetal .qmi4 files")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / expected_import_output_dir(),
        help="Output directory under models/ or agent_workspace/",
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--source-checkpoint", default=None)
    args = parser.parse_args()

    manifest = import_qmi4_artifacts(
        args.source_dir,
        args.out,
        layer_index=args.layer_index,
        source_checkpoint=args.source_checkpoint,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
