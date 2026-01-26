"""Load HuggingFace-style config.json, map to ModelConfig, and quantize on-the-fly.

Supports config variations across Llama, Mistral, Phi, Qwen, Gemma,
and other transformer architectures that use the standard HuggingFace format.

Weight name mapping:
    from converters.safetensors_loader import map_weight_names, load_mapped_safetensors

    # Remap a state dict (auto-detects architecture)
    mapped = map_weight_names(hf_state_dict, model_type="auto", config=config)

    # Or load directly from file with mapping
    mapped = load_mapped_safetensors(Path("model/"), model_type="auto")

Quantization API:
    from converters.safetensors_loader import load_and_quantize

    state_dict, config = load_and_quantize(Path("model/"), quant_type="fp4", group_size=32)

Note:
    This module uses PyTorch tensors. All tensor operations are performed on CPU
    for maximum compatibility. Arrays are converted to torch.Tensor on load.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metal_marlin.inference import ModelConfig

# Activation function aliases found across different model configs.
_ACT_ALIASES: dict[str, str] = {
    "gelu_new": "gelu",
    "gelu_fast": "gelu",
    "gelu_pytorch_tanh": "gelu",
    "quick_gelu": "gelu",
    "silu": "silu",
    "swiglu": "silu",
    "relu": "relu",
}


def _resolve_act(raw: str) -> str:
    """Normalize activation function name to canonical form."""
    return _ACT_ALIASES.get(raw.lower(), raw.lower())


def _resolve_intermediate_size(cfg: dict[str, Any]) -> int:
    """Determine intermediate (FFN) size from various config keys."""
    if "intermediate_size" in cfg:
        return cfg["intermediate_size"]
    # Phi-style: uses inner_dim or n_inner
    if "n_inner" in cfg:
        return cfg["n_inner"]
    if "inner_dim" in cfg:
        return cfg["inner_dim"]
    # Fallback: 4x hidden for standard FFN, ~2.67x for gated (SwiGLU)
    hidden = cfg["hidden_size"]
    act = cfg.get("hidden_act", "silu")
    if act.lower() in ("silu", "swiglu"):
        # Gated MLP: intermediate is typically (8/3)*hidden rounded to multiple of 256
        raw = int(hidden * 8 / 3)
        return ((raw + 255) // 256) * 256
    return hidden * 4


def _resolve_num_kv_heads(cfg: dict[str, Any]) -> int:
    """Extract num_key_value_heads, handling GQA and MQA configs."""
    if "num_key_value_heads" in cfg:
        return cfg["num_key_value_heads"]
    # Falcon uses multi_query flag
    if cfg.get("multi_query", False):
        return 1
    # Some configs use num_kv_heads directly
    if "num_kv_heads" in cfg:
        return cfg["num_kv_heads"]
    # Default: MHA (kv_heads == q_heads)
    return cfg.get("num_attention_heads", 32)


def _resolve_head_dim(cfg: dict[str, Any]) -> int | None:
    """Extract explicit head_dim if specified (e.g., Phi-3)."""
    if "head_dim" in cfg:
        return cfg["head_dim"]
    # Qwen uses kv_channels
    if "kv_channels" in cfg:
        return cfg["kv_channels"]
    return None


def _resolve_rope_theta(cfg: dict[str, Any]) -> float:
    """Extract RoPE base frequency from config."""
    if "rope_theta" in cfg:
        return float(cfg["rope_theta"])
    # Some models nest it under rope_scaling
    rope_scaling = cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict) and "base" in rope_scaling:
        return float(rope_scaling["base"])
    return 10000.0


def _resolve_norm_eps(cfg: dict[str, Any]) -> float:
    """Extract layer norm epsilon from various config keys."""
    if "rms_norm_eps" in cfg:
        return float(cfg["rms_norm_eps"])
    if "layer_norm_eps" in cfg:
        return float(cfg["layer_norm_eps"])
    if "layer_norm_epsilon" in cfg:
        return float(cfg["layer_norm_epsilon"])
    if "norm_eps" in cfg:
        return float(cfg["norm_eps"])
    return 1e-6


def load_config(config_path: Path) -> ModelConfig:
    """Load a HuggingFace config.json and return a ModelConfig.

    Handles config key variations across architectures:
    - Llama/Mistral: standard HF transformer config
    - Phi: n_inner, partial_rotary_factor
    - Qwen: kv_channels
    - Gemma: head_dim explicit
    - Falcon: multi_query, alibi
    - GPT-NeoX/Pythia: rotary_pct, attention_bias variants

    Args:
        config_path: Path to a HuggingFace-style config.json file.

    Returns:
        ModelConfig populated from the parsed configuration.

    Raises:
        FileNotFoundError: If config_path does not exist.
        KeyError: If required fields (vocab_size, hidden_size,
                  num_hidden_layers, num_attention_heads) are missing.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    # Required fields: raise immediately if missing
    vocab_size = cfg["vocab_size"]
    hidden_size = cfg["hidden_size"]
    num_hidden_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg["num_layers"]
    num_attention_heads = cfg.get("num_attention_heads") or cfg.get("n_head") or cfg["num_heads"]

    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=_resolve_intermediate_size(cfg),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=_resolve_num_kv_heads(cfg),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=_resolve_norm_eps(cfg),
        rope_theta=_resolve_rope_theta(cfg),
        hidden_act=_resolve_act(cfg.get("hidden_act", cfg.get("activation_function", "silu"))),
        mlp_bias=cfg.get("mlp_bias", cfg.get("bias", False)),
        attention_bias=cfg.get("attention_bias", cfg.get("bias", False)),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        head_dim=_resolve_head_dim(cfg),
    )


def estimate_memory(
    model_path: Path, group_size: int = 32
) -> dict[str, float]:
    """Estimate memory before loading by parsing safetensors headers.

    Reads only the tensor metadata of each shard to determine shapes without
    loading actual weight data into memory.

    Args:
        model_path: Path to a .safetensors file or directory of shards.
        group_size: Quantization group size (affects scale tensor overhead).

    Returns:
        Dict with fp16_gb, quantized_gb, savings_ratio, and element counts.
    """
    from safetensors import safe_open

    if model_path.is_dir():
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files in {model_path}")
    else:
        shards = [model_path]

    fp16_bytes = 0
    quantizable_elements = 0

    for shard in shards:
        with safe_open(str(shard), framework="numpy") as f:
            for name in f.keys():
                shape = f.get_tensor(name).shape
                numel = 1
                for d in shape:
                    numel *= d

                if "weight" in name and len(shape) == 2:
                    quantizable_elements += numel
                else:
                    fp16_bytes += numel * 2

    # FP4: 4 bits per weight = 0.5 bytes; scales: float16 per group
    quantized_weight_bytes = quantizable_elements // 2
    scale_bytes = (quantizable_elements // group_size) * 2
    total_quantized = fp16_bytes + quantized_weight_bytes + scale_bytes
    total_fp16 = fp16_bytes + quantizable_elements * 2

    return {
        "fp16_bytes": total_fp16,
        "fp16_gb": total_fp16 / (1024**3),
        "quantized_bytes": total_quantized,
        "quantized_gb": total_quantized / (1024**3),
        "savings_ratio": total_fp16 / max(total_quantized, 1),
        "num_weight_elements": quantizable_elements,
        "num_passthrough_bytes": fp16_bytes,
    }


def load_and_quantize(
    model_path: Path,
    quant_type: str = "fp4",
    group_size: int = 32,
) -> tuple[dict[str, torch.Tensor], ModelConfig]:
    """Load safetensors weights and quantize Linear layers on-the-fly.

    Streams tensors from disk, quantizes 2D weight matrices to FP4 or INT4
    with per-group scales, and returns the complete state dict with packed
    weights ready for MarlinLinear layers.

    Non-weight tensors (embeddings, biases, layer norms) pass through as FP16.
    Layers with dimensions incompatible with packing (N not divisible by 8, or
    K not divisible by group_size) are also passed through unquantized.

    Supports both single-file and sharded (model-00001-of-00003.safetensors)
    HuggingFace model layouts.

    Args:
        model_path: Path to .safetensors file or directory containing shards
            and config.json.
        quant_type: "fp4" for E2M1 FP4 or "int4" for asymmetric unsigned INT4.
        group_size: Elements per quantization group. 32 gives better quality,
            128 gives maximum compression.

    Returns:
        (state_dict, config) where state_dict has "{name}.packed" and
        "{name}.scales" for quantized layers, original names for passthrough.
    """
    from safetensors import safe_open
    from tqdm import tqdm

    from metal_marlin.safetensors_loader import (
        _quantize_tensor_fp4,
        _quantize_tensor_int4,
    )

    # Resolve shards
    if model_path.is_dir():
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files in {model_path}")
        config_dir = model_path
    else:
        shards = [model_path]
        config_dir = model_path.parent

    # Memory estimation
    mem = estimate_memory(model_path, group_size)
    print(
        f"Memory: {mem['fp16_gb']:.2f} GB (FP16) -> "
        f"{mem['quantized_gb']:.2f} GB (quantized), "
        f"{mem['savings_ratio']:.1f}x reduction"
    )

    # Load config
    config_json = config_dir / "config.json"
    config = load_config(config_json) if config_json.exists() else ModelConfig()
    config.quant_type = quant_type

    # Count tensors for progress bar
    total_tensors = 0
    for shard in shards:
        with safe_open(str(shard), framework="numpy") as f:
            total_tensors += len(list(f.keys()))

    state_dict: dict[str, torch.Tensor] = {}
    n_quantized = 0

    with tqdm(total=total_tensors, desc="Quantizing weights", unit="tensor") as pbar:
        for shard in shards:
            with safe_open(str(shard), framework="numpy") as f:
                for name in f.keys():
                    tensor_np = f.get_tensor(name)
                    tensor = torch.from_numpy(tensor_np.copy())

                    if "weight" in name and tensor.ndim == 2:
                        K, N = tensor.shape

                        # Skip incompatible dimensions
                        if N % 8 != 0 or K % group_size != 0:
                            state_dict[name] = tensor.to(torch.float16)
                            pbar.update(1)
                            continue

                        if quant_type == "fp4":
                            packed, scales = _quantize_tensor_fp4(
                                tensor, group_size
                            )
                            state_dict[f"{name}.packed"] = _to_torch(packed)
                            state_dict[f"{name}.scales"] = _to_torch(scales)
                        elif quant_type == "int4":
                            packed, scales, zeros = _quantize_tensor_int4(
                                tensor, group_size
                            )
                            state_dict[f"{name}.packed"] = _to_torch(packed)
                            state_dict[f"{name}.scales"] = _to_torch(scales)
                            state_dict[f"{name}.zeros"] = _to_torch(zeros)
                        else:
                            raise ValueError(
                                f"Unknown quant_type: {quant_type!r}. "
                                "Use 'fp4' or 'int4'."
                            )

                        n_quantized += 1
                        pbar.set_postfix_str(f"{name} [{K}x{N}]")
                    else:
                        state_dict[name] = tensor.to(torch.float16)

                    pbar.update(1)

    print(f"Done: {n_quantized} layers quantized, {len(state_dict)} total entries")
    return state_dict, config


def _to_torch(arr: Any) -> torch.Tensor:
    """Convert array to torch.Tensor."""
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(np.asarray(arr).copy())


# ---------------------------------------------------------------------------
# Weight name mapping: HuggingFace -> generic internal names
# ---------------------------------------------------------------------------
#
# Generic internal names (matching MarlinTransformerBlock layout):
#   layers.{i}.self_attn.q_proj.weight
#   layers.{i}.self_attn.k_proj.weight
#   layers.{i}.self_attn.v_proj.weight
#   layers.{i}.self_attn.o_proj.weight
#   layers.{i}.mlp.gate_proj.weight
#   layers.{i}.mlp.up_proj.weight
#   layers.{i}.mlp.down_proj.weight
#   layers.{i}.input_layernorm.weight
#   layers.{i}.post_attention_layernorm.weight
#   embed_tokens.weight
#   lm_head.weight
#   norm.weight  (final RMSNorm)


def _split_fused_qkv(
    fused: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight matrix into separate Q, K, V tensors.

    Handles the concatenated layout:
    [Q: num_heads*head_dim, K: num_kv_heads*head_dim, V: num_kv_heads*head_dim]
    along the output (first) dimension.

    Args:
        fused: Fused weight [total_out, hidden_size]
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head

    Returns:
        (q_weight, k_weight, v_weight)
    """
    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    total = q_size + k_size + k_size  # v_size == k_size

    if fused.shape[0] == total:
        return fused[:q_size], fused[q_size:q_size + k_size], fused[q_size + k_size:]
    elif fused.shape[1] == total:
        return fused[:, :q_size], fused[:, q_size:q_size + k_size], fused[:, q_size + k_size:]

    raise ValueError(
        f"Fused QKV shape {fused.shape} doesn't match expected total "
        f"dimension {total} (num_heads={num_heads}, "
        f"num_kv_heads={num_kv_heads}, head_dim={head_dim})"
    )


def _split_fused_gate_up(
    fused: torch.Tensor,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate_up_proj weight into separate gate and up tensors.

    Layout: [gate: intermediate_size, up: intermediate_size] along output dim.

    Args:
        fused: Fused weight [2*intermediate_size, hidden_size]
        intermediate_size: MLP intermediate dimension

    Returns:
        (gate_weight, up_weight)
    """
    if fused.shape[0] == 2 * intermediate_size:
        return fused[:intermediate_size], fused[intermediate_size:]
    elif fused.shape[1] == 2 * intermediate_size:
        return fused[:, :intermediate_size], fused[:, intermediate_size:]

    raise ValueError(
        f"Fused gate_up shape {fused.shape} doesn't match "
        f"2*intermediate_size={2 * intermediate_size}"
    )


# Architecture-specific mapping rules.
# Each entry: (hf_pattern, generic_name)
# {layer} is placeholder for layer index.
# Prefixes __fused_qkv__, __fused_gate_up__ trigger splitting logic.

_LLAMA_RULES: list[tuple[str, str]] = [
    ("model.embed_tokens.weight", "embed_tokens.weight"),
    ("model.norm.weight", "norm.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("model.layers.{layer}.self_attn.q_proj.weight", "layers.{layer}.self_attn.q_proj.weight"),
    ("model.layers.{layer}.self_attn.k_proj.weight", "layers.{layer}.self_attn.k_proj.weight"),
    ("model.layers.{layer}.self_attn.v_proj.weight", "layers.{layer}.self_attn.v_proj.weight"),
    ("model.layers.{layer}.self_attn.o_proj.weight", "layers.{layer}.self_attn.o_proj.weight"),
    ("model.layers.{layer}.mlp.gate_proj.weight", "layers.{layer}.mlp.gate_proj.weight"),
    ("model.layers.{layer}.mlp.up_proj.weight", "layers.{layer}.mlp.up_proj.weight"),
    ("model.layers.{layer}.mlp.down_proj.weight", "layers.{layer}.mlp.down_proj.weight"),
    ("model.layers.{layer}.input_layernorm.weight", "layers.{layer}.input_layernorm.weight"),
    ("model.layers.{layer}.post_attention_layernorm.weight", "layers.{layer}.post_attention_layernorm.weight"),
    ("model.layers.{layer}.self_attn.q_proj.bias", "layers.{layer}.self_attn.q_proj.bias"),
    ("model.layers.{layer}.self_attn.k_proj.bias", "layers.{layer}.self_attn.k_proj.bias"),
    ("model.layers.{layer}.self_attn.v_proj.bias", "layers.{layer}.self_attn.v_proj.bias"),
    ("model.layers.{layer}.self_attn.o_proj.bias", "layers.{layer}.self_attn.o_proj.bias"),
]

_MISTRAL_RULES = _LLAMA_RULES  # Identical naming convention

_PHI3_RULES: list[tuple[str, str]] = [
    ("model.embed_tokens.weight", "embed_tokens.weight"),
    ("model.final_layernorm.weight", "norm.weight"),
    ("model.final_layernorm.bias", "norm.bias"),
    ("lm_head.weight", "lm_head.weight"),
    ("lm_head.bias", "lm_head.bias"),
    ("model.layers.{layer}.self_attn.qkv_proj.weight", "__fused_qkv__.layers.{layer}.self_attn"),
    ("model.layers.{layer}.self_attn.qkv_proj.bias", "__fused_qkv_bias__.layers.{layer}.self_attn"),
    ("model.layers.{layer}.self_attn.o_proj.weight", "layers.{layer}.self_attn.o_proj.weight"),
    ("model.layers.{layer}.self_attn.o_proj.bias", "layers.{layer}.self_attn.o_proj.bias"),
    ("model.layers.{layer}.mlp.gate_up_proj.weight", "__fused_gate_up__.layers.{layer}.mlp"),
    ("model.layers.{layer}.mlp.gate_up_proj.bias", "__fused_gate_up_bias__.layers.{layer}.mlp"),
    ("model.layers.{layer}.mlp.down_proj.weight", "layers.{layer}.mlp.down_proj.weight"),
    ("model.layers.{layer}.mlp.down_proj.bias", "layers.{layer}.mlp.down_proj.bias"),
    ("model.layers.{layer}.input_layernorm.weight", "layers.{layer}.input_layernorm.weight"),
    ("model.layers.{layer}.input_layernorm.bias", "layers.{layer}.input_layernorm.bias"),
    ("model.layers.{layer}.post_attention_layernorm.weight", "layers.{layer}.post_attention_layernorm.weight"),
    ("model.layers.{layer}.post_attention_layernorm.bias", "layers.{layer}.post_attention_layernorm.bias"),
]

_PHI2_RULES: list[tuple[str, str]] = [
    ("transformer.embd.wte.weight", "embed_tokens.weight"),
    ("lm_head.linear.weight", "lm_head.weight"),
    ("lm_head.linear.bias", "lm_head.bias"),
    ("lm_head.ln.weight", "norm.weight"),
    ("lm_head.ln.bias", "norm.bias"),
    ("transformer.h.{layer}.ln.weight", "layers.{layer}.input_layernorm.weight"),
    ("transformer.h.{layer}.ln.bias", "layers.{layer}.input_layernorm.bias"),
    ("transformer.h.{layer}.mixer.Wqkv.weight", "__fused_qkv__.layers.{layer}.self_attn"),
    ("transformer.h.{layer}.mixer.Wqkv.bias", "__fused_qkv_bias__.layers.{layer}.self_attn"),
    ("transformer.h.{layer}.mixer.out_proj.weight", "layers.{layer}.self_attn.o_proj.weight"),
    ("transformer.h.{layer}.mixer.out_proj.bias", "layers.{layer}.self_attn.o_proj.bias"),
    ("transformer.h.{layer}.mlp.fc1.weight", "__fused_gate_up__.layers.{layer}.mlp"),
    ("transformer.h.{layer}.mlp.fc1.bias", "__fused_gate_up_bias__.layers.{layer}.mlp"),
    ("transformer.h.{layer}.mlp.fc2.weight", "layers.{layer}.mlp.down_proj.weight"),
    ("transformer.h.{layer}.mlp.fc2.bias", "layers.{layer}.mlp.down_proj.bias"),
]

_QWEN2_RULES: list[tuple[str, str]] = [
    ("model.embed_tokens.weight", "embed_tokens.weight"),
    ("model.norm.weight", "norm.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("model.layers.{layer}.self_attn.q_proj.weight", "layers.{layer}.self_attn.q_proj.weight"),
    ("model.layers.{layer}.self_attn.q_proj.bias", "layers.{layer}.self_attn.q_proj.bias"),
    ("model.layers.{layer}.self_attn.k_proj.weight", "layers.{layer}.self_attn.k_proj.weight"),
    ("model.layers.{layer}.self_attn.k_proj.bias", "layers.{layer}.self_attn.k_proj.bias"),
    ("model.layers.{layer}.self_attn.v_proj.weight", "layers.{layer}.self_attn.v_proj.weight"),
    ("model.layers.{layer}.self_attn.v_proj.bias", "layers.{layer}.self_attn.v_proj.bias"),
    ("model.layers.{layer}.self_attn.o_proj.weight", "layers.{layer}.self_attn.o_proj.weight"),
    ("model.layers.{layer}.mlp.gate_proj.weight", "layers.{layer}.mlp.gate_proj.weight"),
    ("model.layers.{layer}.mlp.up_proj.weight", "layers.{layer}.mlp.up_proj.weight"),
    ("model.layers.{layer}.mlp.down_proj.weight", "layers.{layer}.mlp.down_proj.weight"),
    ("model.layers.{layer}.input_layernorm.weight", "layers.{layer}.input_layernorm.weight"),
    ("model.layers.{layer}.post_attention_layernorm.weight", "layers.{layer}.post_attention_layernorm.weight"),
]

_GEMMA_RULES: list[tuple[str, str]] = [
    ("model.embed_tokens.weight", "embed_tokens.weight"),
    ("model.norm.weight", "norm.weight"),
    # Gemma ties embeddings; no separate lm_head
    ("model.layers.{layer}.self_attn.q_proj.weight", "layers.{layer}.self_attn.q_proj.weight"),
    ("model.layers.{layer}.self_attn.k_proj.weight", "layers.{layer}.self_attn.k_proj.weight"),
    ("model.layers.{layer}.self_attn.v_proj.weight", "layers.{layer}.self_attn.v_proj.weight"),
    ("model.layers.{layer}.self_attn.o_proj.weight", "layers.{layer}.self_attn.o_proj.weight"),
    ("model.layers.{layer}.mlp.gate_proj.weight", "layers.{layer}.mlp.gate_proj.weight"),
    ("model.layers.{layer}.mlp.up_proj.weight", "layers.{layer}.mlp.up_proj.weight"),
    ("model.layers.{layer}.mlp.down_proj.weight", "layers.{layer}.mlp.down_proj.weight"),
    ("model.layers.{layer}.input_layernorm.weight", "layers.{layer}.input_layernorm.weight"),
    ("model.layers.{layer}.post_attention_layernorm.weight", "layers.{layer}.post_attention_layernorm.weight"),
]

_STARCODER2_RULES: list[tuple[str, str]] = [
    ("model.embed_tokens.weight", "embed_tokens.weight"),
    ("model.norm.weight", "norm.weight"),
    ("model.norm.bias", "norm.bias"),
    ("lm_head.weight", "lm_head.weight"),
    ("model.layers.{layer}.self_attn.q_proj.weight", "layers.{layer}.self_attn.q_proj.weight"),
    ("model.layers.{layer}.self_attn.q_proj.bias", "layers.{layer}.self_attn.q_proj.bias"),
    ("model.layers.{layer}.self_attn.k_proj.weight", "layers.{layer}.self_attn.k_proj.weight"),
    ("model.layers.{layer}.self_attn.k_proj.bias", "layers.{layer}.self_attn.k_proj.bias"),
    ("model.layers.{layer}.self_attn.v_proj.weight", "layers.{layer}.self_attn.v_proj.weight"),
    ("model.layers.{layer}.self_attn.v_proj.bias", "layers.{layer}.self_attn.v_proj.bias"),
    ("model.layers.{layer}.self_attn.o_proj.weight", "layers.{layer}.self_attn.o_proj.weight"),
    ("model.layers.{layer}.self_attn.o_proj.bias", "layers.{layer}.self_attn.o_proj.bias"),
    ("model.layers.{layer}.mlp.c_fc.weight", "layers.{layer}.mlp.up_proj.weight"),
    ("model.layers.{layer}.mlp.c_fc.bias", "layers.{layer}.mlp.up_proj.bias"),
    ("model.layers.{layer}.mlp.c_proj.weight", "layers.{layer}.mlp.down_proj.weight"),
    ("model.layers.{layer}.mlp.c_proj.bias", "layers.{layer}.mlp.down_proj.bias"),
    ("model.layers.{layer}.input_layernorm.weight", "layers.{layer}.input_layernorm.weight"),
    ("model.layers.{layer}.input_layernorm.bias", "layers.{layer}.input_layernorm.bias"),
    ("model.layers.{layer}.post_attention_layernorm.weight", "layers.{layer}.post_attention_layernorm.weight"),
    ("model.layers.{layer}.post_attention_layernorm.bias", "layers.{layer}.post_attention_layernorm.bias"),
]

_ARCHITECTURE_RULES: dict[str, list[tuple[str, str]]] = {
    "llama": _LLAMA_RULES,
    "mistral": _MISTRAL_RULES,
    "phi": _PHI3_RULES,
    "phi2": _PHI2_RULES,
    "qwen2": _QWEN2_RULES,
    "gemma": _GEMMA_RULES,
    "starcoder2": _STARCODER2_RULES,
}


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a rule pattern with {layer} placeholder to a compiled regex."""
    escaped = re.escape(pattern).replace(r"\{layer\}", r"(?P<layer>\d+)")
    return re.compile(f"^{escaped}$")


def _build_mapper(
    rules: list[tuple[str, str]],
) -> list[tuple[re.Pattern[str], str]]:
    """Compile mapping rules into (regex, target_template) pairs."""
    return [(_pattern_to_regex(src), dst) for src, dst in rules]


def detect_model_type(state_dict_keys: set[str]) -> str:
    """Auto-detect model architecture from weight names.

    Checks for distinguishing patterns unique to each architecture family.

    Args:
        state_dict_keys: Set of weight name strings.

    Returns:
        Architecture identifier matching _ARCHITECTURE_RULES keys.
    """
    # Phi-2: transformer.h.* prefix
    if any(k.startswith("transformer.h.") for k in state_dict_keys):
        return "phi2"

    # Phi-3: has qkv_proj or gate_up_proj fused
    if any("qkv_proj" in k for k in state_dict_keys):
        return "phi"
    if any("gate_up_proj" in k for k in state_dict_keys):
        return "phi"

    # StarCoder2: has c_fc/c_proj in MLP
    if any("mlp.c_fc" in k for k in state_dict_keys):
        return "starcoder2"

    # Qwen2: has biases on Q/K/V projections but not StarCoder
    if any("self_attn.q_proj.bias" in k for k in state_dict_keys):
        return "qwen2"

    # Gemma: no lm_head.weight (tied embeddings)
    has_lm_head = any(k == "lm_head.weight" for k in state_dict_keys)
    if not has_lm_head and any("model.embed_tokens" in k for k in state_dict_keys):
        return "gemma"

    # Default: Llama-style (Mistral uses identical names)
    return "llama"


def map_weight_names(
    hf_state_dict: dict[str, torch.Tensor],
    model_type: str = "auto",
    *,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    intermediate_size: int | None = None,
    config: ModelConfig | dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Map HuggingFace weight names to generic internal layer names.

    Handles renaming, fused QKV splitting, and fused gate_up splitting.
    The output dict uses the naming convention that MarlinTransformerBlock expects.

    Args:
        hf_state_dict: State dict with HuggingFace-convention weight names.
        model_type: Architecture type ("llama", "mistral", "phi", "phi2",
            "qwen2", "gemma", "starcoder2") or "auto" for auto-detection.
        num_heads: Number of query heads (required for fused QKV splitting).
        num_kv_heads: Number of KV heads for GQA (defaults to num_heads).
        head_dim: Dimension per head (inferred from hidden_size/num_heads).
        intermediate_size: MLP intermediate dim (required for fused gate_up,
            inferred from tensor shape if not provided).
        config: ModelConfig or raw config dict. If provided, dimensions are
            extracted from it as defaults.

    Returns:
        New state dict with generic weight names.
    """
    if model_type == "auto":
        model_type = detect_model_type(set(hf_state_dict.keys()))

    if model_type not in _ARCHITECTURE_RULES:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Supported: {list(_ARCHITECTURE_RULES.keys())}"
        )

    # Extract dimensions from config
    if isinstance(config, ModelConfig):
        num_heads = num_heads or config.num_attention_heads
        num_kv_heads = num_kv_heads or config.num_key_value_heads
        head_dim = head_dim or config.head_dim
        intermediate_size = intermediate_size or config.intermediate_size
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
    elif isinstance(config, dict):
        num_heads = num_heads or config.get("num_attention_heads")
        num_kv_heads = num_kv_heads or config.get("num_key_value_heads", num_heads)
        head_dim = head_dim or config.get("head_dim")
        intermediate_size = intermediate_size or config.get("intermediate_size")
        if head_dim is None and num_heads is not None:
            hidden_size = config.get("hidden_size")
            if hidden_size is not None:
                head_dim = hidden_size // num_heads

    num_kv_heads = num_kv_heads or num_heads

    rules = _ARCHITECTURE_RULES[model_type]
    compiled = _build_mapper(rules)
    mapped: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for hf_name, tensor in hf_state_dict.items():
        matched = False

        for regex, target_template in compiled:
            m = regex.match(hf_name)
            if m is None:
                continue

            matched = True
            layer_idx = m.group("layer") if "layer" in m.groupdict() else None
            target = target_template.format(layer=layer_idx) if layer_idx else target_template

            if target.startswith("__fused_qkv__."):
                prefix = target[len("__fused_qkv__."):]
                if num_heads is None or head_dim is None:
                    raise ValueError(
                        f"Fused QKV weight '{hf_name}' requires num_heads and "
                        f"head_dim to split. Provide them directly or via config."
                    )
                q, k, v = _split_fused_qkv(tensor, num_heads, num_kv_heads or num_heads, head_dim)
                mapped[f"{prefix}.q_proj.weight"] = q
                mapped[f"{prefix}.k_proj.weight"] = k
                mapped[f"{prefix}.v_proj.weight"] = v

            elif target.startswith("__fused_qkv_bias__."):
                prefix = target[len("__fused_qkv_bias__."):]
                if num_heads is None or head_dim is None:
                    raise ValueError(
                        f"Fused QKV bias '{hf_name}' requires num_heads and head_dim."
                    )
                q_size = num_heads * head_dim
                kv_size = (num_kv_heads or num_heads) * head_dim
                mapped[f"{prefix}.q_proj.bias"] = tensor[:q_size]
                mapped[f"{prefix}.k_proj.bias"] = tensor[q_size:q_size + kv_size]
                mapped[f"{prefix}.v_proj.bias"] = tensor[q_size + kv_size:]

            elif target.startswith("__fused_gate_up__."):
                prefix = target[len("__fused_gate_up__."):]
                inferred_intermediate = intermediate_size or (tensor.shape[0] // 2)
                gate, up = _split_fused_gate_up(tensor, inferred_intermediate)
                mapped[f"{prefix}.gate_proj.weight"] = gate
                mapped[f"{prefix}.up_proj.weight"] = up

            elif target.startswith("__fused_gate_up_bias__."):
                prefix = target[len("__fused_gate_up_bias__."):]
                mid = intermediate_size or (tensor.shape[0] // 2)
                mapped[f"{prefix}.gate_proj.bias"] = tensor[:mid]
                mapped[f"{prefix}.up_proj.bias"] = tensor[mid:]

            else:
                mapped[target] = tensor

            break

        if not matched:
            unmapped.append(hf_name)
            mapped[hf_name] = tensor

    if unmapped:
        warnings.warn(
            f"map_weight_names: {len(unmapped)} weight(s) not matched by "
            f"{model_type!r} rules, passed through unchanged: "
            f"{unmapped[:5]}{'...' if len(unmapped) > 5 else ''}",
            stacklevel=2,
        )

    return mapped


def load_mapped_safetensors(
    path: str | Path,
    model_type: str = "auto",
    *,
    config: ModelConfig | dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> tuple[dict[str, torch.Tensor], ModelConfig]:
    """Load safetensors file(s) and map weight names to generic convention.

    Combines safetensors loading with weight name mapping. Reads config.json
    from the model directory for automatic dimension inference and returns
    both the mapped state dict and the parsed ModelConfig.

    Args:
        path: Path to .safetensors file or directory with sharded files.
        model_type: Architecture type or "auto" for auto-detection.
        config: Pre-loaded config. If None, loads config.json from path's directory.
        config_path: Explicit path to config.json.

    Returns:
        (mapped_state_dict, model_config) tuple.
    """
    from safetensors import safe_open

    path = Path(path)

    # Resolve config
    if config is None:
        if config_path is not None:
            cfg_path = Path(config_path)
        else:
            cfg_path = (path.parent if path.is_file() else path) / "config.json"

        if cfg_path.exists():
            config = load_config(cfg_path)

    # Load safetensors
    state_dict: dict[str, torch.Tensor] = {}
    if path.is_dir():
        shard_files = sorted(path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")
        for shard in shard_files:
            with safe_open(str(shard), framework="numpy") as f:
                for name in f.keys():
                    state_dict[name] = torch.from_numpy(f.get_tensor(name).copy())
    else:
        with safe_open(str(path), framework="numpy") as f:
            for name in f.keys():
                state_dict[name] = torch.from_numpy(f.get_tensor(name).copy())

    mapped = map_weight_names(state_dict, model_type=model_type, config=config)

    # Ensure we return a ModelConfig
    if not isinstance(config, ModelConfig):
        if isinstance(config, dict):
            config = ModelConfig(
                hidden_size=config.get("hidden_size", 4096),
                num_attention_heads=config.get("num_attention_heads", 32),
                num_key_value_heads=config.get("num_key_value_heads", 32),
                intermediate_size=config.get("intermediate_size", 11008),
            )
        else:
            config = ModelConfig()

    return mapped, config


def get_supported_architectures() -> list[str]:
    """Return list of supported architecture identifiers."""
    return list(_ARCHITECTURE_RULES.keys())


def register_architecture(
    name: str,
    rules: list[tuple[str, str]],
    *,
    overwrite: bool = False,
) -> None:
    """Register custom architecture mapping rules at runtime.

    Args:
        name: Architecture identifier (used as model_type argument).
        rules: List of (hf_pattern, generic_name) tuples.
            Use {layer} for layer index placeholder.
            Use __fused_qkv__ prefix for fused QKV weights.
            Use __fused_gate_up__ prefix for fused gate/up weights.
        overwrite: If True, allow replacing existing rules.
    """
    if name in _ARCHITECTURE_RULES and not overwrite:
        raise ValueError(
            f"Architecture {name!r} already registered. "
            f"Use overwrite=True to replace."
        )
    _ARCHITECTURE_RULES[name] = rules
