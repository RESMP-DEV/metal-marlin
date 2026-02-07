#!/usr/bin/env python3
"""CUDA MR-GPTQ quantization for GLM-4.7-Flash to Marlin-style emulated NVFP4.

This script produces Marlin-compatible FP4 E2M1 packed weights using:
- Bartowski v3 calibration data (default)
- Hadamard + GPTQ (MR-GPTQ)
- CUDA-accelerated quantization backend

Output format:
- Packed FP4 nibbles in uint32 tensors
- Per-group FP16 scales
- `model.safetensors` + `quantization_config.json`
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metal_marlin.calibration import CalibrationDataset
from metal_marlin.hf_loader import download_model
from metal_marlin.mr_gptq import (
    AcceleratedMRGPTQQuantizer,
    MRGPTQQuantizer,
    QuantizationFormat,
)


DEFAULT_MODEL = "zai-org/GLM-4.7-Flash"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "GLM-4.7-Flash-Marlin-NVFP4-CUDA"

COPY_ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


class CUDANVFP4Quantizer(AcceleratedMRGPTQQuantizer):
    """CUDA-accelerated MR-GPTQ quantizer with Marlin scale layout fix."""

    def quantize_layer(
        self,
        weights: np.ndarray,
        hessian: np.ndarray | None = None,
        layer_name: str = "",
        use_hadamard: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if hessian is None:
            return MRGPTQQuantizer.quantize_layer(
                self,
                weights=weights,
                hessian=None,
                layer_name=layer_name,
                use_hadamard=use_hadamard,
            )

        packed, scales, meta = self.quantize_layer_accelerated(
            weights=weights,
            hessian=hessian,
            layer_name=layer_name,
            use_hadamard=use_hadamard,
        )

        out_features = int(weights.shape[0])
        # Accelerated backend emits [n_groups, out_features]. Marlin expects [out_features, n_groups].
        if scales.ndim == 2 and scales.shape[1] == out_features and scales.shape[0] != out_features:
            scales = scales.T.copy()

        if scales.ndim != 2 or scales.shape[0] != out_features:
            raise ValueError(
                f"Unexpected scales shape {scales.shape} for layer {layer_name}, "
                f"expected [out_features, n_groups] with out_features={out_features}"
            )

        return packed, scales, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA Marlin-style emulated NVFP4 quantization for GLM-4.7-Flash",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Local model path or HuggingFace model id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HuggingFace revision if --model is a repo id",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated/private models",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Optional local directory for model download cache",
    )
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument(
        "--hadamard-block-size",
        type=int,
        default=64,
        help="Hadamard block size (power of 2)",
    )
    parser.add_argument("--no-hadamard", action="store_true", help="Disable Hadamard rotation")
    parser.add_argument(
        "--no-actorder",
        action="store_true",
        help="Disable activation-order column permutation",
    )
    parser.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damping ratio")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Limit Bartowski v3 calibration samples (default: all)",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=None,
        help="Number of calibration batches (default: auto from sample count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length for calibration tokenization",
    )
    parser.add_argument(
        "--force-download-calibration",
        action="store_true",
        help="Force re-download of Bartowski v3 calibration file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume support",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output verbosity")
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


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
        )


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
    calibration_count: int,
    calibration_batches: int,
) -> None:
    payload: dict[str, Any] = {
        "format": "fp4_e2m1_emulated_nvfp4",
        "method": "mr_gptq_cuda",
        "model_path": str(model_path),
        "group_size": args.group_size,
        "use_hadamard": not args.no_hadamard,
        "hadamard_block_size": args.hadamard_block_size,
        "actorder": not args.no_actorder,
        "percdamp": args.percdamp,
        "backend": "cuda",
        "calibration": {
            "dataset": "bartowski-v3",
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

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires an NVIDIA GPU.")
        return 1

    if verbose:
        print("=" * 72)
        print("GLM-4.7-Flash MR-GPTQ CUDA Quantization (Marlin emulated NVFP4)")
        print("=" * 72)
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_path = resolve_model_path(
        model_ref=args.model,
        revision=args.revision,
        token=args.hf_token,
        download_dir=args.download_dir,
        verbose=verbose,
    )
    output_path = args.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Model path: {model_path}")
        print(f"Output path: {output_path}")
        print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)

    if verbose:
        print("Loading Bartowski v3 calibration dataset...")
    calibration = CalibrationDataset.v3(
        max_samples=args.calibration_samples,
        force_download=args.force_download_calibration,
    )
    calibration_count = len(calibration)
    calibration_batches = args.calibration_batches
    if calibration_batches is None:
        calibration_batches = math.ceil(calibration_count / max(args.batch_size, 1))

    if verbose:
        print(f"Calibration samples: {calibration_count}")
        print(f"Calibration batches: {calibration_batches}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_seq_len}")

    quantizer = CUDANVFP4Quantizer(
        backend_name="cuda",
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=args.group_size,
        use_hadamard=not args.no_hadamard,
        hadamard_block_size=args.hadamard_block_size,
        actorder=not args.no_actorder,
        percdamp=args.percdamp,
    )

    report = quantizer.quantize_model_with_calibration(
        model_path=model_path,
        calibration=calibration,
        tokenizer=tokenizer,
        output_path=output_path,
        num_calibration_batches=calibration_batches,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        resume=not args.no_resume,
        verbose=verbose,
    )

    copy_model_artifacts(model_path, output_path)
    save_quantization_config(
        output_dir=output_path,
        model_path=model_path,
        report=report,
        args=args,
        calibration_count=calibration_count,
        calibration_batches=calibration_batches,
    )

    if verbose:
        print()
        print("Saved:")
        print(f"  {output_path / 'model.safetensors'}")
        print(f"  {output_path / 'quantization_report.json'}")
        print(f"  {output_path / 'quantization_config.json'}")
        print("Done.")

    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
