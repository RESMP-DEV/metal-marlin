"""Qwen3.6-27B dense hybrid profile.

This module is the tracked, importable source of truth for the Qwen3.6-27B
text-runtime contract used by the fused Metal path.  The generated
``agent_workspace/qwen36_27b/shape_contract.json`` remains evidence, but code
should depend on this profile rather than reading ignored workspace files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
CONFIG_URL = "https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json"
MODEL_CARD_URL = "https://huggingface.co/Qwen/Qwen3.6-27B"
FEATURE_FLAG = "METAL_MARLIN_QWEN36_27B_MEGAKERNEL"


@dataclass(frozen=True)
class Qwen36DeltaNetProfile:
    key_heads: int = 16
    value_heads: int = 48
    key_dim: int = 128
    value_dim: int = 128
    conv_kernel_dim: int = 4

    @property
    def q_features(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def k_features(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def v_features(self) -> int:
        return self.value_heads * self.value_dim

    @property
    def beta_features(self) -> int:
        return self.value_heads

    @property
    def state_elements(self) -> int:
        return self.value_heads * self.key_dim * self.value_dim


@dataclass(frozen=True)
class Qwen36FullAttentionProfile:
    heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    rotary_dim: int = 64

    @property
    def q_features(self) -> int:
        # Qwen3.6 full-attention projection emits Q plus output gate.
        return self.heads * self.head_dim * 2

    @property
    def kv_features(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def o_features(self) -> int:
        return self.heads * self.head_dim


@dataclass(frozen=True)
class Qwen36DenseMLPProfile:
    intermediate_size: int = 17408


@dataclass(frozen=True)
class Qwen36ModelProfile:
    model_id: str = MODEL_ID
    model_type: str = "qwen3_5"
    text_model_type: str = "qwen3_5_text"
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    full_attention_interval: int = 4
    group_size: int = 128
    rms_norm_eps: float = 1e-6
    delta: Qwen36DeltaNetProfile = Qwen36DeltaNetProfile()
    attention: Qwen36FullAttentionProfile = Qwen36FullAttentionProfile()
    dense_mlp: Qwen36DenseMLPProfile = Qwen36DenseMLPProfile()

    @property
    def full_attention_layer_indices(self) -> list[int]:
        return [
            idx
            for idx in range(self.num_hidden_layers)
            if idx % self.full_attention_interval == self.full_attention_interval - 1
        ]

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layer_indices)

    @property
    def num_linear_attention_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_attention_layers

    @property
    def layer_types(self) -> list[str]:
        return [
            "full_attention" if idx in self.full_attention_layer_indices else "linear_attention"
            for idx in range(self.num_hidden_layers)
        ]


QWEN36_27B_PROFILE = Qwen36ModelProfile()


def _unwrap_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def profile_from_hf_config(config: dict[str, Any]) -> Qwen36ModelProfile:
    """Build a Qwen3.6-27B profile from Hugging Face ``config.json`` data."""
    text = _unwrap_text_config(config)
    delta = Qwen36DeltaNetProfile(
        key_heads=int(text["linear_num_key_heads"]),
        value_heads=int(text["linear_num_value_heads"]),
        key_dim=int(text["linear_key_head_dim"]),
        value_dim=int(text["linear_value_head_dim"]),
        conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
    )
    attention = Qwen36FullAttentionProfile(
        heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        rotary_dim=int(text.get("partial_rotary_factor", 0.25) * int(text["head_dim"])),
    )
    dense_mlp = Qwen36DenseMLPProfile(
        intermediate_size=int(text["intermediate_size"]),
    )
    return Qwen36ModelProfile(
        model_id=str(config.get("_name_or_path") or MODEL_ID),
        model_type=str(config.get("model_type", text.get("model_type", "qwen3_5"))),
        text_model_type=str(text.get("model_type", "qwen3_5_text")),
        hidden_size=int(text["hidden_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        vocab_size=int(text["vocab_size"]),
        max_position_embeddings=int(text["max_position_embeddings"]),
        full_attention_interval=int(text["full_attention_interval"]),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        delta=delta,
        attention=attention,
        dense_mlp=dense_mlp,
    )


def load_profile_from_config_file(path: str | Path) -> Qwen36ModelProfile:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return profile_from_hf_config(data)


def shape_contract_payload(profile: Qwen36ModelProfile = QWEN36_27B_PROFILE) -> dict[str, Any]:
    """Return the compact contract payload used by validation scripts."""
    payload = asdict(profile)
    payload.update(
        {
            "model_label": "Qwen 3.6 27B",
            "config_source": CONFIG_URL,
            "model_card": MODEL_CARD_URL,
            "mlp_kind": "dense",
            "weights_loaded": False,
            "layer_types": profile.layer_types,
            "num_full_attention_layers": profile.num_full_attention_layers,
            "num_linear_attention_layers": profile.num_linear_attention_layers,
            "full_attention_layer_indices": profile.full_attention_layer_indices,
        }
    )
    return payload


def is_qwen36_27b_profile(profile: Qwen36ModelProfile) -> bool:
    """Return True when a profile exactly matches the supported fused path."""
    expected = QWEN36_27B_PROFILE
    return (
        profile.hidden_size == expected.hidden_size
        and profile.num_hidden_layers == expected.num_hidden_layers
        and profile.vocab_size == expected.vocab_size
        and profile.full_attention_interval == expected.full_attention_interval
        and profile.dense_mlp.intermediate_size == expected.dense_mlp.intermediate_size
        and profile.delta == expected.delta
        and profile.attention == expected.attention
    )


def is_qwen36_27b_config(config: dict[str, Any], model_name: str = "") -> bool:
    """Detect the dense Qwen3.6-27B config without treating 35B-A3B as equivalent."""
    try:
        profile = profile_from_hf_config(config)
    except (KeyError, TypeError, ValueError):
        return False

    name_blob = f"{model_name} {config.get('_name_or_path', '')}".lower()
    if "qwen3.6-35b" in name_blob or "a3b" in name_blob:
        return False
    if "qwen3.6-27b" in name_blob or profile.hidden_size == 5120:
        return is_qwen36_27b_profile(profile)
    return False


def validate_supported_profile(profile: Qwen36ModelProfile) -> None:
    """Raise a clear error when the fused path cannot support *profile*."""
    if not is_qwen36_27b_profile(profile):
        raise ValueError(
            "Qwen3.6-27B fused path requires dense 27B dimensions: "
            "hidden_size=5120, layers=64, interval=4, dense intermediate=17408, "
            "DeltaNet heads=(16 key, 48 value), full attention=(24 Q, 4 KV, head_dim=256)."
        )

