"""Shared helpers for Qwen3.5 / Qwen3.6 MoE Metal MR-GPTQ quantization scripts.

Architecture constants, artifact-copy lists, weight-map loading, layer-prefix
inference, and quantization-config serialisation are factored here so that both
the 35B-A3B and 122B-A10B scripts share the same fragile parsing code.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from metal_marlin.hf_loader import download_model


def resolve_model_path(
    model_ref: str,
    revision: str = "main",
    token: str | None = None,
    download_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Return a local path, downloading from HuggingFace if necessary."""
    local_path = Path(model_ref)
    if local_path.exists():
        return local_path
    if verbose:
        print(f"Downloading model: {model_ref} (revision={revision})")
    return download_model(
        model_id=model_ref,
        local_dir=download_dir,
        revision=revision,
        token=token,
    )


# ---------------------------------------------------------------------------
# FP4 E2M1 constants – shared with metal_marlin.mr_gptq / quantize_fp4
# ---------------------------------------------------------------------------
FP4_E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------------------------------------------------------------------
# Artifact names shared by all Qwen3.5 / Qwen3.6 checkpoints
# ---------------------------------------------------------------------------
QwenMoE_COPY_ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to both Qwen3.5 and Qwen3.6 quantisation scripts."""
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (elements per scale factor)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "mps"),
        default="mps",
        help="Quantization backend (mps = Metal Performance Shaders on Apple Silicon)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=None)
    parser.add_argument(
        "--hessian-mode",
        choices=("full", "layer_stream"),
        default="layer_stream",
    )
    parser.add_argument(
        "--hessian-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--model-load-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--force-download-calibration", action="store_true")
    parser.add_argument("--quiet", action="store_true")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_weight_map(model_path: Path) -> dict[str, str]:
    """Load the safetensors weight-map index if present."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        return json.load(f).get("weight_map", {})


def infer_layer_prefixes(weight_map: dict[str, str]) -> list[str]:
    """Heuristically find the transformer-layer name prefix."""
    candidates = (
        "model.language_model.layers.",
        "model.layers.",
        "language_model.layers.",
        "transformer.layers.",
        "layers.",
    )
    found: list[str] = []
    for prefix in candidates:
        if any(name.startswith(prefix) for name in weight_map):
            found.append(prefix)
    if not found:
        # Default to the two most common Qwen layouts
        found = list(candidates[:2])
    return found


def copy_model_artifacts(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer / config artefacts from source to output directory."""
    for name in QwenMoE_COPY_ARTIFACTS:
        src = src_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dst_dir / name)


def load_model_config(model_path: Path) -> dict[str, Any]:
    """Load the model config.json as a flat dict, unwrapping ``text_config``."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    # Qwen-style nested config: { "text_config": { ... } }
    text_config = config.get("text_config", {})
    return text_config if text_config else config


def save_quantization_config(
    output_dir: Path,
    model_path: Path,
    report: Any,
    args: argparse.Namespace,
    layer_prefixes: list[str],
    calibration_count: int,
    calibration_batches: int,
    num_experts: int,
    intermediate_dim: int,
    method: str = "mr_gptq_mps",
    extra: dict[str, Any] | None = None,
) -> None:
    """Write ``quantization_config.json`` into the output directory."""
    payload: dict[str, Any] = {
        "format": "mmfp4_e2m1_marlin",
        "method": method,
        "model_id": str(args.model),
        "model_path": str(model_path),
        "group_size": args.group_size,
        "layer_prefixes": layer_prefixes,
        "backend": args.backend,
        "hessian_mode": args.hessian_mode,
        "hessian_dtype": args.hessian_dtype,
        "model_load_dtype": args.model_load_dtype,
        # MoE architecture
        "num_experts": num_experts,
        "intermediate_dim": intermediate_dim,
        "calibration": {
            "enabled": not args.no_calibration,
            "samples": calibration_count,
            "batches": calibration_batches,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
        },
    }
    if extra:
        payload.update(extra)
    if hasattr(report, "to_dict"):
        payload["report"] = report.to_dict()
    with (output_dir / "quantization_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
