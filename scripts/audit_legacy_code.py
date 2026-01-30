#!/usr/bin/env python3
"""
Audit legacy model implementations that should be replaced by
Transformers + layer replacement.
"""

import json
import sys
from pathlib import Path

LEGACY_PATTERNS = [
    "QuantizedLlamaAttention",
    "QuantizedLlamaMLP",
    "QuantizedLlamaLayer",
    "QuantizedQwen3Attention",
    "QuantizedQwen3MLP",
    "QuantizedQwen3Layer",
    "MetalGLM47Model",
    "MetalMLAAttention",
    "MetalMLP",
    "MetalAttention",
    "MixtralAttention",
    "MixtralExpertMLP",
    "MixtralLayer",
]


def find_legacy_code(root: Path) -> dict:
    """Find all legacy model class definitions and usages."""
    results = {
        "definitions": [],  # Class definitions
        "usages": [],  # Imports/instantiations
        "files_affected": set(),
    }

    for py_file in root.rglob("*.py"):
        if ".venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        content = py_file.read_text()

        for pattern in LEGACY_PATTERNS:
            if f"class {pattern}" in content:
                results["definitions"].append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "class": pattern,
                    }
                )
                results["files_affected"].add(str(py_file))

            if pattern in content and f"class {pattern}" not in content:
                # Usage, not definition
                results["usages"].append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "pattern": pattern,
                    }
                )
                results["files_affected"].add(str(py_file))

    results["files_affected"] = list(results["files_affected"])
    return results


def resolve_root(repo_root: Path) -> Path:
    direct = repo_root / "metal_marlin"
    if direct.exists():
        return direct
    contrib = repo_root / "contrib" / "metal_marlin" / "metal_marlin"
    if contrib.exists():
        return contrib
    return direct


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    root = resolve_root(repo_root)
    if not root.exists():
        print(f"Legacy audit root not found: {root}")
        sys.exit(1)

    results = find_legacy_code(root)

    print(f"Found {len(results['definitions'])} legacy class definitions")
    print(f"Found {len(results['usages'])} legacy usages")
    print(f"Files affected: {len(results['files_affected'])}")

    print("\n=== Definitions ===")
    for d in results["definitions"]:
        print(f"  {d['file']}: class {d['class']}")

    print("\n=== Files to Review ===")
    for f in sorted(results["files_affected"]):
        print(f"  {f}")

    # Save for task generation
    with open("legacy_audit.json", "w") as f:
        json.dump(results, f, indent=2)
