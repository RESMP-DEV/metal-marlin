"""Qwen3.6-27B fused-path artifact metadata.

The prototype qwenmetal layout is supported only through this explicit schema:
packed unsigned int4 weights as ``qweight[(K / 8), N]`` plus FP16 per-group
``scales`` and ``zeros`` shaped ``[ceil(K / group_size), N]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .qwen36_27b_profile import MODEL_ID, QWEN36_27B_PROFILE

SCHEMA_VERSION = "qwen36_27b_int4_v1"
TensorRole = Literal[
    "linear_attn_q",
    "linear_attn_k",
    "linear_attn_v",
    "linear_attn_beta",
    "linear_attn_a",
    "linear_attn_z",
    "linear_attn_out",
    "full_attn_q",
    "full_attn_k",
    "full_attn_v",
    "full_attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "lm_head",
]


ROLE_SHAPES: dict[str, tuple[int, int]] = {
    "linear_attn_q": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.q_features,
    ),
    "linear_attn_k": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.k_features,
    ),
    "linear_attn_v": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.v_features,
    ),
    "linear_attn_beta": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.beta_features,
    ),
    "linear_attn_a": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.beta_features,
    ),
    "linear_attn_z": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.v_features,
    ),
    "linear_attn_out": (
        QWEN36_27B_PROFILE.delta.v_features,
        QWEN36_27B_PROFILE.hidden_size,
    ),
    "full_attn_q": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.attention.q_features,
    ),
    "full_attn_k": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.attention.kv_features,
    ),
    "full_attn_v": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.attention.kv_features,
    ),
    "full_attn_o": (
        QWEN36_27B_PROFILE.attention.o_features,
        QWEN36_27B_PROFILE.hidden_size,
    ),
    "mlp_gate": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.dense_mlp.intermediate_size,
    ),
    "mlp_up": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.dense_mlp.intermediate_size,
    ),
    "mlp_down": (
        QWEN36_27B_PROFILE.dense_mlp.intermediate_size,
        QWEN36_27B_PROFILE.hidden_size,
    ),
    "lm_head": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.vocab_size,
    ),
}


@dataclass(frozen=True)
class Int4TensorArtifact:
    role: TensorRole
    layer_index: int | None
    qweight: str
    scales: str
    zeros: str
    in_features: int
    out_features: int
    group_size: int = QWEN36_27B_PROFILE.group_size
    dtype: str = "uint4_asym"
    qweight_layout: str = "packed_k_major_u32"
    metadata_layout: str = "group_major_f16"
    calibration: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.role not in ROLE_SHAPES:
            raise ValueError(f"unsupported Qwen3.6-27B tensor role: {self.role}")
        expected_in, expected_out = ROLE_SHAPES[self.role]
        if (self.in_features, self.out_features) != (expected_in, expected_out):
            raise ValueError(
                f"{self.role} expected shape ({expected_in}, {expected_out}), "
                f"got ({self.in_features}, {self.out_features})"
            )
        if self.in_features % 8 != 0:
            raise ValueError("qweight layout requires in_features divisible by 8")
        if self.group_size <= 0 or self.group_size % 8 != 0:
            raise ValueError("group_size must be a positive multiple of 8")
        if self.layer_index is not None and not 0 <= self.layer_index < QWEN36_27B_PROFILE.num_hidden_layers:
            raise ValueError(f"layer_index out of range: {self.layer_index}")


@dataclass(frozen=True)
class Qwen36ArtifactManifest:
    tensors: list[Int4TensorArtifact]
    schema_version: str = SCHEMA_VERSION
    model_id: str = MODEL_ID
    source_checkpoint: str = MODEL_ID
    notes: str = "Qwen3.6-27B dense fused-path artifact"

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for tensor in self.tensors:
            tensor.validate()


def write_manifest(manifest: Qwen36ArtifactManifest, path: str | Path) -> Path:
    manifest.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> Qwen36ArtifactManifest:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    tensors = [Int4TensorArtifact(**item) for item in payload.get("tensors", [])]
    manifest = Qwen36ArtifactManifest(
        tensors=tensors,
        schema_version=payload.get("schema_version", SCHEMA_VERSION),
        model_id=payload.get("model_id", MODEL_ID),
        source_checkpoint=payload.get("source_checkpoint", MODEL_ID),
        notes=payload.get("notes", ""),
    )
    manifest.validate()
    return manifest


def expected_tensor_shape(role: TensorRole) -> tuple[int, int]:
    return ROLE_SHAPES[role]

