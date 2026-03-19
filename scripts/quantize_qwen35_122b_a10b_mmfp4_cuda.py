#!/usr/bin/env python3
"""CUDA MR-GPTQ quantization for Qwen/Qwen3.5-122B-A10B (MMFP4).

This script targets the Qwen3.5 122B A10B checkpoint and writes a Metal Marlin
MMFP4-compatible output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metal_marlin.calibration import CalibrationDataset
from metal_marlin.hf_loader import download_model
from metal_marlin.mr_gptq import AcceleratedMRGPTQQuantizer, QuantizationFormat

DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "Qwen3.5-122B-A10B-MMFP4-CUDA"
COPY_ARTIFACTS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.5-122B-A10B to MMFP4 with CUDA MR-GPTQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--backend", choices=("auto", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=None)
    parser.add_argument("--hessian-mode", choices=("full", "layer_stream"), default="layer_stream")
    parser.add_argument("--hessian-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--model-load-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--force-download-calibration", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_model_path(
    model_ref: str,
    revision: str,
    token: str | None,
    download_dir: Path | None,
    verbose: bool,
) -> Path:
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


def load_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        return json.load(f).get("weight_map", {})


def infer_layer_prefixes(weight_map: dict[str, str]) -> list[str]:
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
        found = list(candidates[:2])
    return found


def copy_model_artifacts(src_dir: Path, dst_dir: Path) -> None:
    for name in COPY_ARTIFACTS:
        src = src_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dst_dir / name)


def save_quantization_config(
    output_dir: Path,
    model_path: Path,
    report: Any,
    args: argparse.Namespace,
    layer_prefixes: list[str],
    calibration_count: int,
    calibration_batches: int,
) -> None:
    payload: dict[str, Any] = {
        "format": "mmfp4_e2m1_marlin",
        "method": "mr_gptq_cuda",
        "model_id": str(args.model),
        "model_path": str(model_path),
        "group_size": args.group_size,
        "layer_prefixes": layer_prefixes,
        "backend": args.backend,
        "hessian_mode": args.hessian_mode,
        "hessian_dtype": args.hessian_dtype,
        "model_load_dtype": args.model_load_dtype,
        "calibration": {
            "enabled": not args.no_calibration,
            "samples": calibration_count,
            "batches": calibration_batches,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
        },
    }
    if hasattr(report, "to_dict"):
        payload["report"] = report.to_dict()
    with (output_dir / "quantization_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = parse_args()
    verbose = not args.quiet

    if not args.no_calibration and not hasattr(CalibrationDataset, "v3"):
        print("ERROR: CalibrationDataset.v3 is unavailable in this build.")
        return 1

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    model_path = resolve_model_path(
        model_ref=args.model,
        revision=args.revision,
        token=args.hf_token,
        download_dir=args.download_dir,
        verbose=verbose,
    )
    output_path = args.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    weight_map = load_weight_map(model_path)
    layer_prefixes = infer_layer_prefixes(weight_map)

    if verbose:
        print("=" * 72)
        print("Qwen3.5-122B-A10B MR-GPTQ CUDA Quantization (MMFP4)")
        print("=" * 72)
        print(f"Model path: {model_path}")
        print(f"Output path: {output_path}")
        print(f"Layer prefixes: {layer_prefixes}")

    calibration = None
    calibration_count = 0
    calibration_batches = 0
    tokenizer = None
    if not args.no_calibration:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        calibration = CalibrationDataset.v3(
            max_samples=args.calibration_samples,
            force_download=args.force_download_calibration,
        )
        calibration_count = len(calibration)
        calibration_batches = args.calibration_batches or math.ceil(
            calibration_count / max(args.batch_size, 1)
        )
        if verbose:
            print(f"Calibration samples: {calibration_count}")
            print(f"Calibration batches: {calibration_batches}")

    quantizer = AcceleratedMRGPTQQuantizer.create(
        backend=args.backend,
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=args.group_size,
        use_hadamard=True,
        hadamard_block_size=64,
        actorder=True,
        percdamp=0.01,
    )

    if calibration is not None and tokenizer is not None:
        report = quantizer.quantize_model_with_calibration(
            model_path=model_path,
            calibration=calibration,
            tokenizer=tokenizer,
            output_path=output_path,
            layers_to_quantize=layer_prefixes,
            num_calibration_batches=calibration_batches,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            hessian_dtype=args.hessian_dtype,
            hessian_collection_mode=args.hessian_mode,
            model_load_dtype=args.model_load_dtype,
            model_device_map="auto",
            model_low_cpu_mem_usage=True,
            verbose=verbose,
        )
    else:
        report = quantizer.quantize_model(
            model_path=model_path,
            output_path=output_path,
            layers_to_quantize=layer_prefixes,
            verbose=verbose,
        )

    copy_model_artifacts(model_path, output_path)
    save_quantization_config(
        output_dir=output_path,
        model_path=model_path,
        report=report,
        args=args,
        layer_prefixes=layer_prefixes,
        calibration_count=calibration_count,
        calibration_batches=calibration_batches,
    )

    if verbose:
        print("\nDone.")
        print(f"Quantized model written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
