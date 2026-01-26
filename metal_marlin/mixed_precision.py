"""
Mixed-precision quantization for transformer models.

Different layers have vastly different sensitivity to quantization:
- Embeddings, norms, lm_head: Keep FP16 (small relative to model)
- Attention Q/K/V/O: Sensitive to position encoding, use FP8 or tight FP4
- Router/gating (MoE): Critical for expert selection, keep FP16 or FP8
- Expert MLPs (MoE): Redundant (2 of 64 active), aggressive FP4 ok
- MTP heads: Just need "good enough" drafts, aggressive FP4 with large groups

For MoE models like GLM-4.7-Flash:
- 64 experts, 2 active per token → most expert weights are "cold"
- Router accuracy directly impacts quality → keep precise
- Expert diversity matters more than individual expert precision

For MTP (Multi-Token Prediction):
- Auxiliary heads predict N future tokens in parallel
- Verified by main model, so approximation is fine
- Can use larger group sizes (256+) for higher compression
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Precision(Enum):
    """Quantization precision levels."""

    FP16 = "fp16"  # Keep original (16 bits)
    BF16 = "bf16"  # Brain float16 (16 bits, larger dynamic range)
    FP8_E4M3 = "fp8"  # 8-bit float (NVIDIA FP8)
    FP4_E2M1 = "fp4"  # 4-bit float (Marlin MXFP4)
    INT8 = "int8"  # 8-bit integer (per-channel scales)
    INT4 = "int4"  # 4-bit integer (per-group scales)
    # Sub-4-bit quantization for aggressive MoE expert compression
    # Inspired by llama.cpp IQ2_XXS/IQ3_XXS which show 2-3 bit works for cold experts
    INT3 = "int3"  # 3-bit integer (8 levels, 25% smaller than INT4)
    INT2 = "int2"  # 2-bit integer (4 levels, 50% smaller than INT4)
    NF3 = "nf3"  # NormalFloat 3-bit (Gaussian quantiles, better for Gaussian weights)
    NF2 = "nf2"  # NormalFloat 2-bit (Gaussian quantiles, most aggressive)


@dataclass
class LayerQuantConfig:
    """Quantization config for a specific layer pattern."""

    precision: Precision = Precision.FP4_E2M1
    group_size: int = 128
    symmetric: bool = True  # For INT4/INT8


@dataclass
class MixedPrecisionConfig:
    """
    Mixed-precision quantization configuration.

    Defines precision levels for different layer types based on sensitivity.
    """

    # Default for unmatched layers
    default: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )

    # Critical layers - keep high precision (BF16 preferred for larger dynamic range)
    embeddings: LayerQuantConfig = field(default_factory=lambda: LayerQuantConfig(Precision.BF16))
    lm_head: LayerQuantConfig = field(default_factory=lambda: LayerQuantConfig(Precision.BF16))
    norms: LayerQuantConfig = field(default_factory=lambda: LayerQuantConfig(Precision.BF16))

    # Attention layers - position-sensitive
    attention_qkv: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 64)
    )
    attention_out: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 64)
    )

    # MLP layers (dense models)
    mlp_gate: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )
    mlp_up: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )
    mlp_down: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )

    # MoE-specific layers
    moe_router: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.BF16)  # Critical for expert selection
    )
    moe_experts: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 128)
    )
    moe_shared_expert: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 64)  # More used
    )

    # MTP (Multi-Token Prediction) heads
    mtp_heads: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig(Precision.FP4_E2M1, 256)  # Aggressive
    )

    @classmethod
    def default_dense(cls) -> MixedPrecisionConfig:
        """Config for standard dense transformer (Llama, Mistral, etc.)."""
        return cls()

    @classmethod
    def default_moe(cls) -> MixedPrecisionConfig:
        """Config optimized for MoE models (Mixtral, GLM-4.7-Flash, etc.)."""
        return cls(
            # Router is critical - determines which experts fire (BF16 for dynamic range)
            moe_router=LayerQuantConfig(Precision.BF16),
            # Shared expert sees all tokens, keep tighter
            moe_shared_expert=LayerQuantConfig(Precision.FP4_E2M1, 64),
            # Routed experts can be more aggressive (redundancy)
            moe_experts=LayerQuantConfig(Precision.FP4_E2M1, 128),
            # Attention still sensitive
            attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 64),
        )

    @classmethod
    def aggressive_moe(cls) -> MixedPrecisionConfig:
        """
        Aggressive MoE quantization using sub-4-bit for cold experts.

        For MoE models with 64 experts, only 2 active per token:
        - 62 "cold" experts can be quantized to INT3/NF3
        - Community benchmarks (llama.cpp IQ2_XXS, IQ3_XXS) show this works
        - Shared expert stays at INT4/FP4 since it's always used

        Use NF3 (NormalFloat 3-bit) for cold experts as transformer weights
        are approximately Gaussian distributed.
        """
        return cls(
            # Router still critical
            moe_router=LayerQuantConfig(Precision.BF16),
            # Shared expert used every token - keep at INT4
            moe_shared_expert=LayerQuantConfig(Precision.INT4, 64),
            # Cold routed experts - aggressive NF3 (Gaussian-optimal)
            moe_experts=LayerQuantConfig(Precision.NF3, 64),
            # Attention still sensitive
            attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 64),
        )

    @classmethod
    def extreme_moe(cls) -> MixedPrecisionConfig:
        """
        Extreme MoE quantization using INT2/NF2 for cold experts.

        Maximum compression for deployment on memory-constrained devices.
        Use with caution - quality may degrade on complex tasks.
        """
        return cls(
            # Router is sacred
            moe_router=LayerQuantConfig(Precision.BF16),
            # Shared expert at INT4
            moe_shared_expert=LayerQuantConfig(Precision.INT4, 64),
            # Cold experts at NF2 (most aggressive, Gaussian-optimal)
            moe_experts=LayerQuantConfig(Precision.NF2, 64),
            # Attention at FP4 with small groups
            attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 32),
        )

    @classmethod
    def default_moe_mtp(cls) -> MixedPrecisionConfig:
        """Config for MoE + MTP models like GLM-4.7-Flash."""
        return cls(
            # Router precision is absolute priority (BF16 for large attention values)
            moe_router=LayerQuantConfig(Precision.BF16),
            # Shared expert handles base capability
            moe_shared_expert=LayerQuantConfig(Precision.FP4_E2M1, 64),
            # Routed experts are redundant
            moe_experts=LayerQuantConfig(Precision.FP4_E2M1, 128),
            # MTP heads just need to be "good enough" for draft verification
            mtp_heads=LayerQuantConfig(Precision.FP4_E2M1, 256),
            # Attention stays tight
            attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 64),
        )

    @classmethod
    def quality_first(cls) -> MixedPrecisionConfig:
        """Prioritize quality over compression."""
        return cls(
            default=LayerQuantConfig(Precision.FP4_E2M1, 64),
            attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 32),
            attention_out=LayerQuantConfig(Precision.FP4_E2M1, 32),
            moe_router=LayerQuantConfig(Precision.BF16),
            moe_shared_expert=LayerQuantConfig(Precision.FP4_E2M1, 32),
            moe_experts=LayerQuantConfig(Precision.FP4_E2M1, 64),
            mtp_heads=LayerQuantConfig(Precision.FP4_E2M1, 128),
        )

    @classmethod
    def speed_first(cls) -> MixedPrecisionConfig:
        """Prioritize speed/compression over quality."""
        return cls(
            default=LayerQuantConfig(Precision.FP4_E2M1, 256),
            attention_qkv=LayerQuantConfig(Precision.FP4_E2M1, 128),
            attention_out=LayerQuantConfig(Precision.FP4_E2M1, 128),
            mlp_gate=LayerQuantConfig(Precision.FP4_E2M1, 256),
            mlp_up=LayerQuantConfig(Precision.FP4_E2M1, 256),
            mlp_down=LayerQuantConfig(Precision.FP4_E2M1, 256),
            moe_router=LayerQuantConfig(Precision.BF16),  # Still critical for expert selection
            moe_shared_expert=LayerQuantConfig(Precision.FP4_E2M1, 128),
            moe_experts=LayerQuantConfig(Precision.FP4_E2M1, 256),
            mtp_heads=LayerQuantConfig(Precision.FP4_E2M1, 512),
        )


# Layer name pattern matching
LAYER_PATTERNS = {
    # Order matters - more specific patterns first
    "embeddings": [
        "embed_tokens",
        "wte",
        "word_embedding",
        "embedding",
    ],
    "lm_head": [
        "lm_head",
        "output.weight",
        "output_layer",
    ],
    "norms": [
        "layernorm",
        "layer_norm",
        "rmsnorm",
        "rms_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "final_layernorm",
        "norm",
    ],
    # MoE patterns (check before regular MLP)
    "moe_router": [
        "router",
        "gate.weight",
        "moe_gate",
        "expert_gate",
        "block_sparse_moe.gate",
    ],
    "moe_shared_expert": [
        "shared_expert",
        "shared_experts",
    ],
    "moe_experts": [
        "experts.",
        "expert.",
        "block_sparse_moe.experts",
        "moe.experts",
    ],
    # MTP patterns
    "mtp_heads": [
        "mtp_head",
        "mtp_predictor",
        "multi_token",
        "auxiliary_head",
        "draft_head",
    ],
    # Attention patterns
    "attention_qkv": [
        "q_proj",
        "k_proj",
        "v_proj",
        "query",
        "key",
        "value",
        "qkv_proj",
        "c_attn",
    ],
    "attention_out": [
        "o_proj",
        "out_proj",
        "dense",
        "c_proj",
    ],
    # MLP patterns (dense models)
    "mlp_gate": [
        "gate_proj",
        "w1",
        "fc1",
        "c_fc",
    ],
    "mlp_up": [
        "up_proj",
        "w3",
        "fc2",
    ],
    "mlp_down": [
        "down_proj",
        "w2",
        "fc_out",
        "c_proj",
    ],
}


def classify_layer(name: str) -> str:
    """
    Classify a layer name into a category.

    Returns the category key (e.g., 'attention_qkv', 'moe_router') or 'default'.
    """
    name_lower = name.lower()

    for category, patterns in LAYER_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return category

    return "default"


def get_layer_config(
    name: str,
    config: MixedPrecisionConfig,
) -> LayerQuantConfig:
    """Get quantization config for a specific layer."""
    category = classify_layer(name)
    return getattr(config, category, config.default)


def should_quantize(
    name: str,
    tensor: np.ndarray,
    config: MixedPrecisionConfig,
) -> tuple[bool, LayerQuantConfig]:
    """
    Determine if/how to quantize a tensor.

    Returns:
        (should_quantize, config) - False if precision is FP16 or tensor not suitable
    """
    layer_config = get_layer_config(name, config)

    # FP16/BF16 means no quantization (just precision format)
    if layer_config.precision in (Precision.FP16, Precision.BF16):
        return False, layer_config

    # Must be 2D weight matrix
    if tensor.ndim != 2:
        return False, layer_config

    # Check dimension compatibility
    out_feat, in_feat = tensor.shape

    # For FP4, need divisibility by 8 (packing)
    if layer_config.precision in (Precision.FP4_E2M1, Precision.INT4):
        if in_feat % 8 != 0:
            return False, layer_config

    # For group size, need divisibility
    if in_feat % layer_config.group_size != 0:
        # Find compatible group size
        for gs in [256, 128, 64, 32, 16, 8]:
            if gs <= layer_config.group_size and in_feat % gs == 0:
                layer_config = LayerQuantConfig(
                    precision=layer_config.precision,
                    group_size=gs,
                    symmetric=layer_config.symmetric,
                )
                break
        else:
            return False, layer_config

    return True, layer_config


# ============================================================================
# Quantization dispatch
# ============================================================================


def quantize_tensor(
    tensor: np.ndarray,
    config: LayerQuantConfig,
) -> dict[str, np.ndarray]:
    """
    Quantize a tensor according to config.

    Returns dict with packed data and metadata (scales, zeros, etc.)
    """
    from .quantize_fp4 import quantize_fp4

    if config.precision == Precision.FP16:
        return {"data": tensor.astype(np.float16)}

    elif config.precision == Precision.BF16:
        # BF16 uses same memory as FP16 but with 8-bit exponent (same as FP32)
        # NumPy doesn't have native bf16, so we store as uint16 with bf16 flag
        # The actual conversion happens in MLX/Metal at runtime
        import mlx.core as mx

        # Convert via MLX which has native bf16 support
        mlx_tensor = mx.array(tensor.astype(np.float32))
        bf16_tensor = mlx_tensor.astype(mx.bfloat16)
        # Store raw bits as uint16 for numpy compatibility
        return {"data": np.array(bf16_tensor).view(np.uint16), "dtype": "bfloat16"}

    elif config.precision == Precision.FP4_E2M1:
        packed, scales = quantize_fp4(tensor, group_size=config.group_size)
        return {
            "packed": packed,
            "scales": scales,
            "group_size": np.array([config.group_size], dtype=np.int32),
            "precision": np.array([4], dtype=np.int32),  # 4 bits
        }

    elif config.precision == Precision.INT4:
        # INT4 with zero point
        packed, scales, zeros = quantize_int4(tensor, config.group_size, config.symmetric)
        result = {
            "packed": packed,
            "scales": scales,
            "group_size": np.array([config.group_size], dtype=np.int32),
            "precision": np.array([4], dtype=np.int32),
        }
        if zeros is not None:
            result["zeros"] = zeros
        return result

    elif config.precision == Precision.FP8_E4M3:
        # FP8 quantization
        return quantize_fp8(tensor, config.group_size)

    elif config.precision == Precision.INT8:
        # INT8 per-channel quantization
        return quantize_int8(tensor, config.symmetric)

    # Sub-4-bit quantization formats
    elif config.precision == Precision.INT2:
        from .sub4bit import quantize_int2
        packed, scales = quantize_int2(tensor, group_size=config.group_size)
        return {
            "packed": packed,
            "scales": scales,
            "group_size": np.array([config.group_size], dtype=np.int32),
            "precision": np.array([2], dtype=np.int32),
            "quant_type": "int2",
        }

    elif config.precision == Precision.INT3:
        from .sub4bit import quantize_int3
        packed, scales = quantize_int3(tensor, group_size=config.group_size)
        return {
            "packed": packed,
            "scales": scales,
            "group_size": np.array([config.group_size], dtype=np.int32),
            "precision": np.array([3], dtype=np.int32),
            "quant_type": "int3",
        }

    elif config.precision == Precision.NF2:
        from .sub4bit import quantize_nf2
        packed, scales = quantize_nf2(tensor, group_size=config.group_size)
        return {
            "packed": packed,
            "scales": scales,
            "group_size": np.array([config.group_size], dtype=np.int32),
            "precision": np.array([2], dtype=np.int32),
            "quant_type": "nf2",
        }

    elif config.precision == Precision.NF3:
        from .sub4bit import quantize_nf3
        packed, scales = quantize_nf3(tensor, group_size=config.group_size)
        return {
            "packed": packed,
            "scales": scales,
            "group_size": np.array([config.group_size], dtype=np.int32),
            "precision": np.array([3], dtype=np.int32),
            "quant_type": "nf3",
        }

    else:
        raise ValueError(f"Unsupported precision: {config.precision}")


def quantize_int4(
    tensor: np.ndarray,
    group_size: int = 128,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Quantize tensor to INT4.

    Args:
        tensor: FP16/FP32 tensor [out_features, in_features]
        group_size: Elements per quantization group
        symmetric: If True, use symmetric quantization (no zero point)

    Returns:
        (packed_uint32, scales, zeros or None)
    """
    out_feat, in_feat = tensor.shape
    num_groups = in_feat // group_size

    # Reshape for per-group processing
    t = tensor.reshape(out_feat, num_groups, group_size).astype(np.float32)

    if symmetric:
        # Symmetric: scale = max(|x|) / 7
        abs_max = np.abs(t).max(axis=2, keepdims=True) + 1e-10
        scales = (abs_max / 7.0).astype(np.float16).squeeze(2)

        # Quantize: round(x / scale), clip to [-8, 7]
        quantized = np.clip(np.round(t / abs_max * 7), -8, 7).astype(np.int8)
        zeros = None
    else:
        # Asymmetric: scale = (max - min) / 15, zero = -min / scale
        t_min = t.min(axis=2, keepdims=True)
        t_max = t.max(axis=2, keepdims=True)

        scales = ((t_max - t_min) / 15.0 + 1e-10).astype(np.float16).squeeze(2)
        zeros = (-t_min / (scales[..., None] + 1e-10)).astype(np.float16).squeeze(2)

        # Quantize: round((x - min) / scale), clip to [0, 15]
        quantized = np.clip(np.round((t - t_min) / (t_max - t_min + 1e-10) * 15), 0, 15).astype(
            np.uint8
        )

    # Pack 8 INT4 values into uint32
    quantized = quantized.reshape(out_feat, in_feat)
    if symmetric:
        # Shift signed [-8,7] to unsigned [0,15]
        quantized = (quantized + 8).astype(np.uint8)

    packed = np.zeros((out_feat, in_feat // 8), dtype=np.uint32)
    for i in range(8):
        packed |= quantized[:, i::8].astype(np.uint32) << (i * 4)

    return packed, scales, zeros


def quantize_fp8(
    tensor: np.ndarray,
    group_size: int = 128,
) -> dict[str, np.ndarray]:
    """
    Quantize tensor to FP8 E4M3.

    E4M3 format: sign(1) | exp(4, bias=7) | mantissa(3)
    Range: [-448, 448], precision: 3 mantissa bits
    """
    out_feat, in_feat = tensor.shape
    num_groups = in_feat // group_size

    # Reshape for per-group processing
    t = tensor.reshape(out_feat, num_groups, group_size).astype(np.float32)

    # Per-group scale: max value maps to 448 (FP8 E4M3 max)
    abs_max = np.abs(t).max(axis=2, keepdims=True) + 1e-10
    scales = (abs_max / 448.0).astype(np.float16).squeeze(2)

    # Scale and clip
    scaled = t / (scales[:, :, None] + 1e-10)
    scaled = np.clip(scaled, -448, 448)

    # For simplicity, store as int8 with separate scale
    # (True FP8 would use bit manipulation)
    # Map [-448, 448] to [-127, 127]
    quantized = np.clip(np.round(scaled * (127 / 448)), -127, 127).astype(np.int8)
    quantized = quantized.reshape(out_feat, in_feat)

    return {
        "data": quantized,
        "scales": scales,
        "group_size": np.array([group_size], dtype=np.int32),
        "precision": np.array([8], dtype=np.int32),
    }


def quantize_int8(
    tensor: np.ndarray,
    symmetric: bool = True,
) -> dict[str, np.ndarray]:
    """
    Quantize tensor to INT8 (per-channel).

    Per-channel quantization uses one scale per output row.
    """
    out_feat, in_feat = tensor.shape
    t = tensor.astype(np.float32)

    if symmetric:
        # Per-channel scale
        abs_max = np.abs(t).max(axis=1, keepdims=True) + 1e-10
        scales = (abs_max / 127.0).astype(np.float16).squeeze(1)

        quantized = np.clip(np.round(t / abs_max * 127), -128, 127).astype(np.int8)
        return {
            "data": quantized,
            "scales": scales,
            "precision": np.array([8], dtype=np.int32),
        }
    else:
        t_min = t.min(axis=1, keepdims=True)
        t_max = t.max(axis=1, keepdims=True)

        scales = ((t_max - t_min) / 255.0 + 1e-10).astype(np.float16).squeeze(1)
        zeros = (-t_min / (scales[:, None] + 1e-10)).astype(np.float16).squeeze(1)

        quantized = np.clip(np.round((t - t_min) / (t_max - t_min + 1e-10) * 255), 0, 255).astype(
            np.uint8
        )

        return {
            "data": quantized,
            "scales": scales,
            "zeros": zeros,
            "precision": np.array([8], dtype=np.int32),
        }


# ============================================================================
# Analysis utilities
# ============================================================================


def analyze_model_layers(
    model_path: str,
    config: MixedPrecisionConfig | None = None,
) -> dict[str, Any]:
    """
    Analyze a model's layers and report quantization plan.

    Returns statistics on how many parameters go to each precision level.
    """
    from .hf_loader import iter_safetensors_weights

    if config is None:
        config = MixedPrecisionConfig.default_dense()

    stats = {
        "total_params": 0,
        "by_precision": {p.value: 0 for p in Precision},
        "by_category": {},
        "layers": [],
    }

    for name, tensor, _ in iter_safetensors_weights(model_path):
        if "weight" not in name.lower():
            continue

        params = tensor.size
        category = classify_layer(name)
        should_q, layer_cfg = should_quantize(name, tensor, config)

        precision = layer_cfg.precision if should_q else Precision.FP16

        stats["total_params"] += params
        stats["by_precision"][precision.value] += params
        stats["by_category"][category] = stats["by_category"].get(category, 0) + params

        stats["layers"].append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "params": params,
                "category": category,
                "precision": precision.value,
                "group_size": layer_cfg.group_size if should_q else None,
            }
        )

    # Compute percentages
    total = stats["total_params"]
    stats["by_precision_pct"] = {k: v / total * 100 for k, v in stats["by_precision"].items()}
    stats["by_category_pct"] = {k: v / total * 100 for k, v in stats["by_category"].items()}

    return stats


def print_analysis(stats: dict[str, Any]) -> None:
    """Pretty-print model analysis."""
    total = stats["total_params"]
    print(f"\nTotal parameters: {total / 1e9:.2f}B")

    print("\nBy precision:")
    for prec, pct in sorted(stats["by_precision_pct"].items(), key=lambda x: -x[1]):
        if pct > 0.1:
            count = stats["by_precision"][prec]
            print(f"  {prec:8s}: {pct:5.1f}% ({count / 1e6:.1f}M params)")

    print("\nBy category:")
    for cat, pct in sorted(stats["by_category_pct"].items(), key=lambda x: -x[1]):
        if pct > 0.1:
            count = stats["by_category"][cat]
            print(f"  {cat:20s}: {pct:5.1f}% ({count / 1e6:.1f}M params)")

    # Estimate compression
    bits_original = total * 16
    bits_quant = sum(
        stats["by_precision"][p.value] * bits
        for p, bits in [
            (Precision.FP16, 16),
            (Precision.BF16, 16),
            (Precision.FP8_E4M3, 8),
            (Precision.FP4_E2M1, 4),
            (Precision.INT8, 8),
            (Precision.INT4, 4),
            (Precision.INT3, 3),
            (Precision.INT2, 2),
            (Precision.NF3, 3),
            (Precision.NF2, 2),
        ]
    )
    print(f"\nEstimated compression: {bits_original / bits_quant:.2f}x")


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mixed-precision analysis and conversion")
    subparsers = parser.add_subparsers(dest="command")

    # Analyze command
    analyze_p = subparsers.add_parser("analyze", help="Analyze model layer sensitivity")
    analyze_p.add_argument("model_path", help="Model path or HF ID")
    analyze_p.add_argument(
        "--preset",
        choices=["dense", "moe", "moe-mtp", "moe-aggressive", "moe-extreme", "quality", "speed"],
        default="dense",
    )

    # Convert command
    convert_p = subparsers.add_parser("convert", help="Convert with mixed precision")
    convert_p.add_argument("model_path", help="Model path or HF ID")
    convert_p.add_argument("output_dir", help="Output directory")
    convert_p.add_argument(
        "--preset",
        choices=["dense", "moe", "moe-mtp", "moe-aggressive", "moe-extreme", "quality", "speed"],
        default="dense",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        preset_map = {
            "dense": MixedPrecisionConfig.default_dense,
            "moe": MixedPrecisionConfig.default_moe,
            "moe-mtp": MixedPrecisionConfig.default_moe_mtp,
            "moe-aggressive": MixedPrecisionConfig.aggressive_moe,
            "moe-extreme": MixedPrecisionConfig.extreme_moe,
            "quality": MixedPrecisionConfig.quality_first,
            "speed": MixedPrecisionConfig.speed_first,
        }
        config = preset_map[args.preset]()
        stats = analyze_model_layers(args.model_path, config)
        print_analysis(stats)

    elif args.command == "convert":
        # TODO: Implement mixed-precision conversion
        print("Mixed-precision conversion not yet implemented")
        print("Use: python -m metal_marlin.hf_loader convert ...")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
