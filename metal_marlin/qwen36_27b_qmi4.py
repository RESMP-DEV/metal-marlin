"""Qwen3.6-27B qwenmetal int4 artifact import helpers."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen36_27b_artifact import (
    Int4TensorArtifact,
    Qwen36ArtifactManifest,
    TensorRole,
    expected_tensor_shape,
    write_manifest,
)
from .qwen36_27b_profile import MODEL_ID

QMI4_MAGIC = b"QMI4v1\0\0"
QMI4_VERSION = 2
QMI4_HEADER = struct.Struct("<8s9I3QI")

LEGACY_TENSOR_ROLES: dict[str, TensorRole] = {
    "q": "linear_attn_q",
    "k": "linear_attn_k",
    "v": "linear_attn_v",
    "b": "linear_attn_beta",
    "beta": "linear_attn_beta",
    "a": "linear_attn_a",
    "z": "linear_attn_z",
    "out": "linear_attn_out",
    "attn_q": "full_attn_q",
    "attn_k": "full_attn_k",
    "attn_v": "full_attn_v",
    "attn_o": "full_attn_o",
    "mlp_gate": "mlp_gate",
    "mlp_up": "mlp_up",
    "mlp_down": "mlp_down",
    "lm_head": "lm_head",
}


@dataclass(frozen=True)
class Qmi4Tensor:
    """In-memory view of a qwenmetal `.qmi4` tensor."""

    path: Path
    source_key: str
    in_features: int
    out_features: int
    group_size: int
    packed_k: int
    groups: int
    source_out_features: int
    row_begin: int
    row_count: int
    qweight: bytes
    scales: bytes
    zeros: bytes

    @property
    def qweight_shape(self) -> tuple[int, int]:
        return self.packed_k, self.out_features

    @property
    def metadata_shape(self) -> tuple[int, int]:
        return self.groups, self.out_features

    def validate(self) -> None:
        if self.group_size <= 0 or self.group_size % 8 != 0:
            raise ValueError(f"group_size must be a positive multiple of 8, got {self.group_size}")
        if self.packed_k != (self.in_features + 7) // 8:
            raise ValueError(
                f"packed_k mismatch: expected {(self.in_features + 7) // 8}, got {self.packed_k}"
            )
        if self.groups != (self.in_features + self.group_size - 1) // self.group_size:
            raise ValueError(
                "group count mismatch: "
                f"expected {(self.in_features + self.group_size - 1) // self.group_size}, "
                f"got {self.groups}"
            )
        expected_qweight = self.packed_k * self.out_features * 4
        expected_metadata = self.groups * self.out_features * 2
        if len(self.qweight) != expected_qweight:
            raise ValueError(f"qweight byte count mismatch: {len(self.qweight)} != {expected_qweight}")
        if len(self.scales) != expected_metadata:
            raise ValueError(f"scales byte count mismatch: {len(self.scales)} != {expected_metadata}")
        if len(self.zeros) != expected_metadata:
            raise ValueError(f"zeros byte count mismatch: {len(self.zeros)} != {expected_metadata}")


def read_qmi4_tensor(path: str | Path) -> Qmi4Tensor:
    """Read a qwenmetal `.qmi4` tensor without depending on the old C++ runtime."""

    source = Path(path)
    data = source.read_bytes()
    if len(data) < QMI4_HEADER.size:
        raise ValueError(f"{source} is too small to be a qmi4 tensor")
    (
        magic,
        version,
        in_features,
        out_features,
        group_size,
        packed_k,
        groups,
        source_out_features,
        row_begin,
        row_count,
        qweight_bytes,
        scales_bytes,
        zeros_bytes,
        key_len,
    ) = QMI4_HEADER.unpack_from(data)
    if magic != QMI4_MAGIC:
        raise ValueError(f"{source} has unsupported qmi4 magic {magic!r}")
    if version != QMI4_VERSION:
        raise ValueError(f"{source} has unsupported qmi4 version {version}")

    offset = QMI4_HEADER.size
    key_end = offset + key_len
    qweight_end = key_end + qweight_bytes
    scales_end = qweight_end + scales_bytes
    zeros_end = scales_end + zeros_bytes
    if zeros_end != len(data):
        raise ValueError(
            f"{source} byte count mismatch: header ends at {zeros_end}, file has {len(data)} bytes"
        )
    source_key = data[offset:key_end].decode("utf-8")
    tensor = Qmi4Tensor(
        path=source,
        source_key=source_key,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        packed_k=packed_k,
        groups=groups,
        source_out_features=source_out_features,
        row_begin=row_begin,
        row_count=row_count,
        qweight=data[key_end:qweight_end],
        scales=data[qweight_end:scales_end],
        zeros=data[scales_end:zeros_end],
    )
    tensor.validate()
    return tensor


def role_from_legacy_name(name: str) -> TensorRole:
    stem = Path(name).stem
    try:
        return LEGACY_TENSOR_ROLES[stem]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen3.6-27B legacy tensor name: {name}") from exc


def write_tensor_parts(tensor: Qmi4Tensor, output_dir: str | Path, prefix: str) -> tuple[str, str, str]:
    """Write qweight/scales/zeros as raw files and return manifest-relative paths."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    qweight = f"{prefix}.qweight.u32.bin"
    scales = f"{prefix}.scales.f16.bin"
    zeros = f"{prefix}.zeros.f16.bin"
    (target / qweight).write_bytes(tensor.qweight)
    (target / scales).write_bytes(tensor.scales)
    (target / zeros).write_bytes(tensor.zeros)
    return qweight, scales, zeros


def _prototype_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def _calibration_metadata(
    prototype: dict[str, Any],
    legacy_name: str,
    tensor: Qmi4Tensor,
    source_path: Path,
) -> dict[str, Any]:
    tensors = prototype.get("tensors", {}) if isinstance(prototype.get("tensors"), dict) else {}
    tensor_meta = tensors.get(legacy_name) or tensors.get(Path(legacy_name).stem) or {}
    quantization = prototype.get("quantization", {})
    calibration = prototype.get("calibration", {})
    return {
        "prototype_format": prototype.get("format", "qwenmetal-qmi4"),
        "prototype_tensor_name": legacy_name,
        "source_tensor": tensor_meta.get("source", tensor.source_key),
        "source_qmi4": str(source_path),
        "row_begin": tensor.row_begin,
        "row_count": tensor.row_count,
        "source_out_features": tensor.source_out_features,
        "method": quantization.get("method", "unknown"),
        "bits": quantization.get("bits", 4),
        "layout": quantization.get("layout", "qweight_packed_u4_k8_by_output_column"),
        "calibration": calibration,
    }


def import_qmi4_artifacts(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    layer_index: int = 0,
    source_checkpoint: str | None = None,
) -> Path:
    """Convert prototype qwenmetal `.qmi4` outputs into the Metal Marlin schema."""

    source = Path(source_dir)
    target = Path(output_dir)
    prototype = _prototype_manifest(source)
    tensors: list[Int4TensorArtifact] = []
    model = source_checkpoint or prototype.get("model") or MODEL_ID

    qmi4_files = sorted(source.glob("*.qmi4"))
    if not qmi4_files:
        raise ValueError(f"no .qmi4 files found in {source}")

    for qmi4_path in qmi4_files:
        legacy_name = qmi4_path.stem
        role = role_from_legacy_name(legacy_name)
        tensor = read_qmi4_tensor(qmi4_path)
        expected_in, expected_out = expected_tensor_shape(role)
        if (tensor.in_features, tensor.out_features) != (expected_in, expected_out):
            raise ValueError(
                f"{qmi4_path.name} maps to {role} but has shape "
                f"({tensor.in_features}, {tensor.out_features}); expected "
                f"({expected_in}, {expected_out})"
            )
        prefix = f"layer{layer_index}.{role}"
        qweight, scales, zeros = write_tensor_parts(tensor, target, prefix)
        tensors.append(
            Int4TensorArtifact(
                role=role,
                layer_index=layer_index,
                qweight=qweight,
                scales=scales,
                zeros=zeros,
                in_features=tensor.in_features,
                out_features=tensor.out_features,
                group_size=tensor.group_size,
                calibration=_calibration_metadata(prototype, legacy_name, tensor, qmi4_path),
            )
        )

    manifest = Qwen36ArtifactManifest(
        tensors=tensors,
        source_checkpoint=model,
        notes=(
            "Imported from qwenmetal qmi4 files; raw tensors are split into "
            "qweight/scales/zeros for Metal Marlin fused Qwen3.6-27B kernels."
        ),
    )
    target.mkdir(parents=True, exist_ok=True)
    return write_manifest(manifest, target / "manifest.json")


def expected_import_output_dir() -> Path:
    return Path("agent_workspace") / "qwen36_27b" / "artifacts" / "layer0"
