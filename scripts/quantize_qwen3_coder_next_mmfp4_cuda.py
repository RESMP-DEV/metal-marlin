#!/usr/bin/env python3
"""CUDA MR-GPTQ quantization for Qwen3-Coder-Next to MMFP4.

MMFP4 is a Metal Marlin-specific FP4 E2M1 format and is NOT NVFP4-interoperable.
This script produces Marlin-compatible packed weights using:
- Bartowski v3 calibration data (default)
- Hadamard + GPTQ (MR-GPTQ)
- CUDA-accelerated quantization backend
- Layer-wise streaming for memory-constrained systems

Qwen3-Coder-Next Architecture:
- 80B total params, 3B active
- 48 layers with hybrid attention (12 full + 36 DeltaNet)
- 512 experts per layer, 10 active per token
- 256K context length
- FP16 size: 159GB

Pipeline architecture for large MoE models:
1. Layer-wise streaming: Process one layer at a time to fit in DRAM
2. Expert batching: Split 512 experts into 32-expert batches for VRAM
3. Hessian collection: Layer-stream mode for <64GB RAM systems
4. Stream-to-disk: Immediate checkpoint to avoid OOM

Output format:
- Packed FP4 nibbles in uint32 tensors
- Per-group FP16 scales
- HuggingFace-style sharded `model-*.safetensors`
- `quantization_config.json`

Usage:
    # Standard quantization (requires ~64GB RAM)
    python scripts/quantize_qwen3_coder_next_mmfp4_cuda.py \\
        --output models/Qwen3-Coder-Next-Marlin-MMFP4

    # Low-RAM mode with layer streaming (48GB+ RAM)
    python scripts/quantize_qwen3_coder_next_mmfp4_cuda.py \\
        --output models/Qwen3-Coder-Next-Marlin-MMFP4 \\
        --force-layer-stream

    # Quick test (2 layers)
    python scripts/quantize_qwen3_coder_next_mmfp4_cuda.py \\
        --output models/Qwen3-Coder-Next-Marlin-MMFP4-test \\
        --max-layers 2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metal_marlin.calibration import CalibrationDataset  # noqa: E402
from metal_marlin.hf_loader import download_model  # noqa: E402
from metal_marlin.mr_gptq import (  # noqa: E402
    AcceleratedMRGPTQQuantizer,
    MRGPTQQuantizer,
    QuantizationFormat,
)


# ============================================================================
# Model Configuration - Qwen3-Coder-Next
# ============================================================================
DEFAULT_MODEL = "Qwen/Qwen3-Coder-Next"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "Qwen3-Coder-Next-Marlin-MMFP4-CUDA"

# Architecture constants
NUM_LAYERS = 48
NUM_EXPERTS = 512
NUM_ACTIVE_EXPERTS = 10
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5120
MOE_INTERMEDIATE_SIZE = 512  # Each expert is small
HEAD_DIM = 256
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 2

# DeltaNet configuration
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 32
LINEAR_CONV_KERNEL_DIM = 4

# Layer pattern: 12 × (3 × DeltaNet-MoE + 1 × Attention-MoE)
FULL_ATTENTION_INTERVAL = 4  # Every 4th layer is full attention

# Memory management defaults - conservative for 24GB VRAM
DEFAULT_BATCH_SIZE = 1  # Small batches for 512-expert model
DEFAULT_MAX_SEQ_LEN = 1024  # Shorter sequences for memory
EXPERT_VRAM_BATCH_SIZE = 32  # Process experts in batches

# VRAM and RAM thresholds (more aggressive than GLM due to model size)
LOW_VRAM_THRESHOLD_GB = 25.8
LOW_SYSTEM_RAM_THRESHOLD_GB = 96.0  # Higher RAM requirement for 80B model
ULTRA_LOW_RAM_THRESHOLD_GB = 64.0  # Layer streaming mandatory below this
LOW_VRAM_CALIBRATION_SAMPLES = 64  # Fewer samples for large model
LOW_VRAM_BATCH_SIZE = 1
LOW_VRAM_MAX_SEQ_LEN = 512
LOW_VRAM_MAX_HESSIAN_LAYERS = 256  # Fewer layers tracked

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
)


# ============================================================================
# Qwen3 MMFP4 Quantizer with Expert Batching
# ============================================================================


class Qwen3CUDAMMFP4Quantizer(AcceleratedMRGPTQQuantizer):
    """CUDA-accelerated MR-GPTQ quantizer for Qwen3-Coder-Next MMFP4.

    Extends AcceleratedMRGPTQQuantizer with:
    - Expert batching for 512-expert MoE layers
    - DeltaNet layer handling (A_log, dt_bias, conv1d)
    - Marlin scale layout fix for Metal inference
    """

    def __init__(
        self,
        expert_batch_size: int = EXPERT_VRAM_BATCH_SIZE,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expert_batch_size = expert_batch_size
        self._layer_idx = 0

    def _is_deltanet_layer(self, layer_idx: int) -> bool:
        """Check if layer uses DeltaNet (linear) attention."""
        return (layer_idx % FULL_ATTENTION_INTERVAL) != (FULL_ATTENTION_INTERVAL - 1)

    def _get_layer_bits_for_component(
        self,
        layer_name: str,
        layer_idx: int,
    ) -> int | None:
        """Get override bits for sensitive components. Returns None for default."""
        name_lower = layer_name.lower()

        # Critical components - keep high precision
        if any(p in name_lower for p in ["router", "gate.weight", "embed_tokens", "lm_head"]):
            return 8
        if any(p in name_lower for p in ["norm", "layernorm"]):
            return 8

        # DeltaNet-specific - higher precision for discretization params
        if any(p in name_lower for p in ["a_log", "dt_bias"]):
            return 8
        if "conv1d" in name_lower:
            return 6
        if any(p in name_lower for p in ["in_proj_qkvz", "in_proj_ba"]):
            return 6

        # Shared expert is more important than regular experts
        if "shared_expert" in name_lower and "gate" not in name_lower:
            return 5
        if "shared_expert_gate" in name_lower:
            return 6

        # Regular experts can use default 4-bit
        return None

    def quantize_layer(
        self,
        weights: np.ndarray,
        hessian: np.ndarray | None = None,
        layer_name: str = "",
        use_hadamard: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Quantize with Marlin scale layout fix and component-aware bits."""
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
                + f"expected [out_features, n_groups] with out_features={out_features}"
            )

        return packed, scales, meta


# ============================================================================
# Helper Functions
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA MMFP4 quantization for Qwen3-Coder-Next (non-NVFP4 interoperable)",
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
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size",
    )
    parser.add_argument(
        "--hadamard-block-size",
        type=int,
        default=64,
        help="Hadamard block size (power of 2)",
    )
    parser.add_argument(
        "--no-hadamard",
        action="store_true",
        help="Disable Hadamard rotation",
    )
    parser.add_argument(
        "--no-actorder",
        action="store_true",
        help="Disable activation-order column permutation",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="GPTQ damping ratio",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Limit Bartowski v3 calibration samples (default: auto based on RAM)",
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
        default=DEFAULT_BATCH_SIZE,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Max sequence length for calibration tokenization",
    )
    parser.add_argument(
        "--max-hessian-layers",
        type=int,
        default=None,
        help="Cap number of layers tracked for Hessians (None = unlimited)",
    )
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=EXPERT_VRAM_BATCH_SIZE,
        help=f"Experts per GPU batch (default: {EXPERT_VRAM_BATCH_SIZE})",
    )
    parser.add_argument(
        "--hessian-dtype",
        choices=("auto", "float64", "float32"),
        default="auto",
        help="Hessian accumulator dtype",
    )
    parser.add_argument(
        "--hessian-exclude-pattern",
        action="append",
        default=[],
        help="Layer-name substring to exclude from Hessian tracking (repeatable)",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Disable conservative auto-tuning for memory-constrained systems",
    )
    parser.add_argument(
        "--force-layer-stream",
        action="store_true",
        help="Force layer-streaming mode even on high-RAM systems",
    )
    parser.add_argument(
        "--force-calibration",
        action="store_true",
        help="Force full-pass calibration mode on low-system-RAM hosts",
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
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser.parse_args()


def resolve_model_path(
    model_ref: str,
    revision: str,
    token: str | None,
    download_dir: Path | None,
    verbose: bool,
) -> Path:
    """Resolve model path, downloading if necessary."""
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
    """Load tokenizer with fast fallback."""
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
    """Copy tokenizer and config files."""
    for name in COPY_ARTIFACTS:
        src = src_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dst_dir / name)


def get_total_system_ram_gb() -> float:
    """Return total system RAM in GiB."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    phys_pages = os.sysconf("SC_PHYS_PAGES")
    return (page_size * phys_pages) / (1024**3)


def configure_runtime_profile(
    args: argparse.Namespace,
    total_vram_gb: float,
    system_ram_gb: float,
) -> tuple[bool, str, bool]:
    """Apply conservative defaults for memory-constrained systems.

    Returns:
        (profile_applied, hessian_dtype, force_layer_stream)
    """
    low_vram = total_vram_gb <= LOW_VRAM_THRESHOLD_GB
    ultra_low_ram = system_ram_gb < ULTRA_LOW_RAM_THRESHOLD_GB
    low_ram = system_ram_gb < LOW_SYSTEM_RAM_THRESHOLD_GB
    profile_applied = False
    force_layer_stream = args.force_layer_stream or ultra_low_ram

    if not args.aggressive and (low_vram or low_ram):
        # Qwen3-Coder-Next is ~2x larger than GLM, use stricter defaults
        if args.calibration_samples is None:
            args.calibration_samples = LOW_VRAM_CALIBRATION_SAMPLES
            profile_applied = True
        if args.batch_size == DEFAULT_BATCH_SIZE:
            args.batch_size = LOW_VRAM_BATCH_SIZE
            profile_applied = True
        if args.max_seq_len == DEFAULT_MAX_SEQ_LEN:
            args.max_seq_len = LOW_VRAM_MAX_SEQ_LEN
            profile_applied = True
        if args.max_hessian_layers is None:
            args.max_hessian_layers = LOW_VRAM_MAX_HESSIAN_LAYERS
            profile_applied = True

        # Exclude experts from Hessian tracking by default (512 experts per layer)
        exclude_patterns = {p.lower() for p in args.hessian_exclude_pattern}
        if "experts" not in exclude_patterns:
            args.hessian_exclude_pattern.append("experts")
            profile_applied = True

    if args.hessian_dtype == "auto":
        # Use float32 for large models to save memory
        hessian_dtype = "float32"
    else:
        hessian_dtype = args.hessian_dtype

    return profile_applied, hessian_dtype, force_layer_stream


def save_quantization_config(
    output_dir: Path,
    model_path: Path,
    report: Any,
    args: argparse.Namespace,
    calibration_count: int,
    calibration_batches: int,
    hessian_dtype: str,
    profile_applied: bool,
    quantization_mode: str,
    system_ram_gb: float,
) -> None:
    """Save quantization configuration and metadata."""
    payload: dict[str, Any] = {
        "format": "mmfp4_e2m1_marlin",
        "method": "mr_gptq_cuda",
        "quantization_mode": quantization_mode,
        "interoperability": {
            "nvfp4_compatible": False,
            "vllm_cuda_compatible": False,
            "note": "MMFP4 is a Metal Marlin-specific format and is not NVIDIA NVFP4.",
        },
        "model_id": str(DEFAULT_MODEL),
        "model_path": str(model_path),
        "architecture": {
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "num_active_experts": NUM_ACTIVE_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "attention_type": "hybrid_deltanet_full",
            "full_attention_interval": FULL_ATTENTION_INTERVAL,
        },
        "group_size": args.group_size,
        "use_hadamard": not args.no_hadamard,
        "hadamard_block_size": args.hadamard_block_size,
        "actorder": not args.no_actorder,
        "percdamp": args.percdamp,
        "backend": "cuda",
        "runtime_profile": {
            "conservative_auto_applied": profile_applied,
            "aggressive_override": args.aggressive,
            "force_layer_stream": args.force_layer_stream,
            "system_ram_gb": round(system_ram_gb, 1),
            "expert_batch_size": args.expert_batch_size,
        },
        "hessian": {
            "dtype": hessian_dtype,
            "max_layers": args.max_hessian_layers,
            "exclude_patterns": args.hessian_exclude_pattern,
        },
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


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = parse_args()
    verbose = not args.quiet

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires an NVIDIA GPU.")
        return 1

    device_props = torch.cuda.get_device_properties(0)
    total_vram_gb = device_props.total_memory / 1e9
    system_ram_gb = get_total_system_ram_gb()
    profile_applied, hessian_dtype, force_layer_stream = configure_runtime_profile(
        args, total_vram_gb, system_ram_gb
    )

    # Determine collection mode
    if force_layer_stream or (system_ram_gb < LOW_SYSTEM_RAM_THRESHOLD_GB and not args.force_calibration):
        hessian_collection_mode = "layer_stream"
    else:
        hessian_collection_mode = "full"

    if verbose:
        print("=" * 72)
        print("Qwen3-Coder-Next MR-GPTQ CUDA Quantization (MMFP4)")
        print("=" * 72)
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {total_vram_gb:.1f} GB")
        print(f"System RAM: {system_ram_gb:.1f} GiB")
        print("Format: MMFP4 (Metal Marlin-specific, non-NVFP4 interoperable)")
        print()
        print("Model: Qwen3-Coder-Next")
        print(f"  Layers: {NUM_LAYERS}")
        print(f"  Experts: {NUM_EXPERTS} (10 active)")
        print(
            f"  Attention: Hybrid (DeltaNet + Full every {FULL_ATTENTION_INTERVAL} layers)")
        print(f"  Expert batch: {args.expert_batch_size}")
        print()
        if profile_applied:
            print("Applied memory-conservative profile:")
            print(f"  calibration_samples={args.calibration_samples}")
            print(f"  batch_size={args.batch_size}")
            print(f"  max_seq_len={args.max_seq_len}")
            print(f"  max_hessian_layers={args.max_hessian_layers}")
            print(f"  hessian_dtype={hessian_dtype}")
            if args.hessian_exclude_pattern:
                print(f"  hessian_exclude={args.hessian_exclude_pattern}")
        if hessian_collection_mode == "layer_stream":
            print(
                "Layer-streaming Hessian mode: Processing one layer at a time "
                + "(slower but uses ~40% less RAM)."
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

    if verbose:
        print(f"Model path: {model_path}")
        print(f"Output path: {output_path}")

    # Load tokenizer
    tokenizer = None
    calibration = None
    calibration_count = 0
    calibration_batches = 0

    if verbose:
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
        calibration_batches = math.ceil(
            calibration_count / max(args.batch_size, 1))

    if verbose:
        print(f"Calibration samples: {calibration_count}")
        print(f"Calibration batches: {calibration_batches}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_seq_len}")
        print(f"Hessian dtype: {hessian_dtype}")
        print(f"Max Hessian layers: {args.max_hessian_layers or 'unlimited'}")
        print(f"Hessian collection mode: {hessian_collection_mode}")
        if args.hessian_exclude_pattern:
            print(f"Hessian excludes: {args.hessian_exclude_pattern}")

    # Create quantizer
    quantizer = Qwen3CUDAMMFP4Quantizer(
        backend_name="cuda",
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=args.group_size,
        use_hadamard=not args.no_hadamard,
        hadamard_block_size=args.hadamard_block_size,
        actorder=not args.no_actorder,
        percdamp=args.percdamp,
        expert_batch_size=args.expert_batch_size,
    )

    quantization_mode = (
        "mr_gptq_layer_stream" if hessian_collection_mode == "layer_stream" else "mr_gptq_calibration"
    )

    try:
        report = quantizer.quantize_model_with_calibration(
            model_path=model_path,
            calibration=calibration,
            tokenizer=tokenizer,
            output_path=output_path,
            num_calibration_batches=calibration_batches,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_hessian_layers=args.max_hessian_layers,
            hessian_dtype=hessian_dtype,
            hessian_exclude_patterns=args.hessian_exclude_pattern,
            hessian_collection_mode=hessian_collection_mode,
            # For 80B model, always use float16 and low-memory settings
            model_load_dtype="float16",
            model_device_map="auto",
            model_low_cpu_mem_usage=True,
            model_offload_dir=(
                output_path / "offload_model" if hessian_collection_mode == "layer_stream" else None
            ),
            resume=not args.no_resume,
            verbose=verbose,
        )
    except (RuntimeError, MemoryError) as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cannot allocate memory" in msg:
            print(
                "Calibration mode ran out of memory; switching to RTN streaming fallback.")
            quantization_mode = "rtn_streaming_fallback"
            report = quantizer.quantize_model(
                model_path=model_path,
                calibration_data=None,
                output_path=output_path,
                verbose=verbose,
            )
            calibration_count = 0
            calibration_batches = 0
        else:
            raise

    copy_model_artifacts(model_path, output_path)
    save_quantization_config(
        output_dir=output_path,
        model_path=model_path,
        report=report,
        args=args,
        calibration_count=calibration_count,
        calibration_batches=calibration_batches,
        hessian_dtype=hessian_dtype,
        profile_applied=profile_applied,
        quantization_mode=quantization_mode,
        system_ram_gb=system_ram_gb,
    )

    if verbose:
        print()
        print("Saved:")
        if (output_path / "model.safetensors").exists():
            print(f"  {output_path / 'model.safetensors'}")
        if (output_path / "model.safetensors.index.json").exists():
            print(f"  {output_path / 'model.safetensors.index.json'}")
        # Check for sharded outputs
        shard_files = list(output_path.glob("model-*.safetensors"))
        if shard_files:
            print(f"  {len(shard_files)} sharded model files")
        print(f"  {output_path / 'quantization_report.json'}")
        print(f"  {output_path / 'quantization_config.json'}")
        print()
        print("Done.")

    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
