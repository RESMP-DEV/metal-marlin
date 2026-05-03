#!/usr/bin/env python3
"""Metal MR-GPTQ quantization for Qwen/Qwen3.6-35B-A3B (MMFP4).

This script targets the Qwen3.6 35B-A3B Mixture-of-Experts checkpoint and
writes a Metal Marlin MMFP4-compatible output directory.  It is the Metal
(Apple Silicon) counterpart of ``quantize_qwen36_122b_a10b_mmfp4_cuda.py``.

Architecture highlights (35B-A3B):
  - 256 routed experts per MoE layer
  - 512 intermediate (FFN) dimension per expert
  - Expert MLP weights stored as 3-D tensors ``[E, out, in]``
  - Quantized with group-128 FP4 E2M1 (MMFP4) and packed 8-per-uint32

Qwen3.6 adds DeltaNet (sparse local attention) layers alongside MoE.
DeltaNet projection layers are kept in full precision.

MMFP4 format:
  - FP4 E2M1 representable grid with LUT-based dequantization
  - Per-group (128 elements) FP16 scale factors
  - 8 FP4 nibbles packed into a single uint32:
        bits [3:0]   = value 0,  bits [7:4]   = value 1,  ...,
        bits [31:28] = value 7
  - Expert packed shape: ``[E, out, in // 8]``
  - Expert scales shape:  ``[E, out, in // group_size]``

Usage:
    cd contrib/metal_marlin
    uv run python scripts/quantize_qwen36_35b_a3b_mmfp4.py \\
        --model Qwen/Qwen3.6-35B-A3B --output models/Qwen3.6-35B-A3B-MMFP4

    # Quick test (no calibration, RTN fallback)
    uv run python scripts/quantize_qwen36_35b_a3b_mmfp4.py --no-calibration
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – allow importing from the metal_marlin package one level up.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Metal / MPS availability check
# ---------------------------------------------------------------------------
try:
    import torch

    if not torch.backends.mps.is_available():
        print(
            "WARNING: MPS not available.  This script targets Apple Silicon "
            "(M1/M2/M3/M4).  Metal kernels will not run.\n"
            "For NVIDIA GPUs use quantize_qwen36_122b_a10b_mmfp4_cuda.py instead.",
            file=sys.stderr,
        )
except ImportError:
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Metal Marlin imports
# ---------------------------------------------------------------------------
from metal_marlin.calibration import CalibrationDataset
from metal_marlin.mr_gptq import AcceleratedMRGPTQQuantizer, QuantizationFormat

# ---------------------------------------------------------------------------
# Shared helpers (Qwen3.5 / Qwen3.6 MoE)
# ---------------------------------------------------------------------------
from scripts._qwen_moe_shared import (
    add_shared_args,
    copy_model_artifacts,
    infer_layer_prefixes,
    load_model_config,
    load_weight_map,
    resolve_model_path,
    save_quantization_config,
)


logger = logging.getLogger(__name__)

# =====================================================================
# Model constants – Qwen3.6-35B-A3B
# =====================================================================
# The 35B-A3B variant is shared by Qwen3.5 and Qwen3.6: both use 256 routed
# experts with 512-dim intermediate FFN per expert.
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "Qwen3.6-35B-A3B-MMFP4"

# Expert topology
NUM_EXPERTS = 256
INTERMEDIATE_DIM = 512

# Quantization hyper-parameters
GROUP_SIZE = 128
HADAMARD_BLOCK_SIZE = 64


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args() -> argparse.Namespace:
    logger.debug("parse_args called")
    parser = argparse.ArgumentParser(
        description=(
            "Quantize Qwen3.6-35B-A3B to MMFP4 (FP4 E2M1) with "
            "Metal-accelerated MR-GPTQ."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_args(parser)
    parser.set_defaults(
        model=DEFAULT_MODEL,
        output=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    logger.info("main starting")
    args = parse_args()
    verbose = not args.quiet

    if not args.no_calibration and not hasattr(CalibrationDataset, "v3"):
        print("ERROR: CalibrationDataset.v3 is unavailable in this build.")
        return 1

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

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
    model_config = load_model_config(model_path)

    if verbose:
        print("=" * 72)
        print("Qwen3.6-35B-A3B MR-GPTQ Metal Quantization (MMFP4)")
        print("=" * 72)
        print(f"Model path:         {model_path}")
        print(f"Output path:        {output_path}")
        print(f"Layer prefixes:     {layer_prefixes}")
        print(f"Experts:            {NUM_EXPERTS}")
        print(f"Intermediate dim:   {INTERMEDIATE_DIM}")
        print(f"Group size:         {args.group_size}")
        print(f"Backend:            {args.backend}")
        print(f"DeltaNet enabled:   {model_config.get('use_delta', False)}")
        print(
            f"DeltaNet int dim:    {model_config.get('delta_intermediate_size', 'N/A')}"
        )
        if model_config.get("layer_types"):
            print(f"Layer types:        {model_config['layer_types'][:5]}...")

    # ------------------------------------------------------------------
    # Calibration setup
    # ------------------------------------------------------------------
    calibration = None
    calibration_count = 0
    calibration_batches = 0
    tokenizer = None

    if not args.no_calibration:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
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

    # ------------------------------------------------------------------
    # Create Metal-backed quantizer
    # ------------------------------------------------------------------
    quantizer = AcceleratedMRGPTQQuantizer.create(
        backend=args.backend,
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=args.group_size,
        use_hadamard=True,
        hadamard_block_size=HADAMARD_BLOCK_SIZE,
        actorder=True,
        percdamp=0.01,
    )

    if verbose:
        print(f"\nQuantizer: {type(quantizer).__name__}")

    # ------------------------------------------------------------------
    # Run quantization
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Finalize output
    # ------------------------------------------------------------------
    copy_model_artifacts(model_path, output_path)
    save_quantization_config(
        output_dir=output_path,
        model_path=model_path,
        report=report,
        args=args,
        layer_prefixes=layer_prefixes,
        calibration_count=calibration_count,
        calibration_batches=calibration_batches,
        num_experts=NUM_EXPERTS,
        intermediate_dim=INTERMEDIATE_DIM,
        method="mr_gptq_mps",
        extra={
            "deltanet": {
                "use_delta": model_config.get("use_delta", False),
                "delta_intermediate_size": model_config.get(
                    "delta_intermediate_size"
                ),
                "layer_types": model_config.get("layer_types", []),
                "full_attention_interval": model_config.get(
                    "full_attention_interval"
                ),
            }
        },
    )

    if verbose:
        print("\nDone.")
        print(f"Quantized model written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
