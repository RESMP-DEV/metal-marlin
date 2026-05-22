from __future__ import annotations

import json
from pathlib import Path

import pytest

from metal_marlin.kernels.qwen36_27b import load_packed_int4_matrix
from metal_marlin.qwen36_27b_artifact import read_manifest
from metal_marlin.qwen36_27b_qmi4 import (
    QMI4_HEADER,
    QMI4_MAGIC,
    QMI4_VERSION,
    import_qmi4_artifacts,
    read_qmi4_tensor,
    role_from_legacy_name,
)


def _write_qmi4(
    path: Path,
    *,
    key: str = "model.language_model.layers.0.linear_attn.in_proj_b.weight",
    in_features: int = 5120,
    out_features: int = 48,
    group_size: int = 128,
) -> None:
    packed_k = (in_features + 7) // 8
    groups = (in_features + group_size - 1) // group_size
    qweight = bytes(packed_k * out_features * 4)
    scales = bytes(groups * out_features * 2)
    zeros = bytes(groups * out_features * 2)
    key_bytes = key.encode("utf-8")
    path.write_bytes(
        QMI4_HEADER.pack(
            QMI4_MAGIC,
            QMI4_VERSION,
            in_features,
            out_features,
            group_size,
            packed_k,
            groups,
            out_features,
            0,
            out_features,
            len(qweight),
            len(scales),
            len(zeros),
            len(key_bytes),
        )
        + key_bytes
        + qweight
        + scales
        + zeros
    )


def test_read_qmi4_tensor_header_and_payload(tmp_path: Path) -> None:
    qmi4 = tmp_path / "b.qmi4"
    _write_qmi4(qmi4)

    tensor = read_qmi4_tensor(qmi4)

    assert tensor.source_key == "model.language_model.layers.0.linear_attn.in_proj_b.weight"
    assert tensor.qweight_shape == (640, 48)
    assert tensor.metadata_shape == (40, 48)
    assert len(tensor.qweight) == 640 * 48 * 4
    assert len(tensor.scales) == 40 * 48 * 2
    assert len(tensor.zeros) == 40 * 48 * 2


def test_read_qmi4_tensor_rejects_bad_magic(tmp_path: Path) -> None:
    qmi4 = tmp_path / "b.qmi4"
    _write_qmi4(qmi4)
    payload = bytearray(qmi4.read_bytes())
    payload[:8] = b"BADMAGIC"
    qmi4.write_bytes(payload)

    with pytest.raises(ValueError, match="unsupported qmi4 magic"):
        read_qmi4_tensor(qmi4)


def test_import_qmi4_artifacts_writes_metal_marlin_manifest(tmp_path: Path) -> None:
    source = tmp_path / "prototype"
    target = tmp_path / "metal_marlin"
    source.mkdir()
    _write_qmi4(source / "b.qmi4")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "format": "qwenmetal-awq-layer0-v1",
                "model": "Qwen/Qwen3.6-27B",
                "quantization": {
                    "method": "awq_activation_weighted",
                    "bits": 4,
                    "layout": "qweight_packed_u4_k8_by_output_column",
                },
                "calibration": {"samples": 8192},
                "tensors": {
                    "b": {
                        "source": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                        "row_begin": 0,
                        "row_count": 48,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = import_qmi4_artifacts(source, target)
    manifest = read_manifest(manifest_path)

    assert manifest.source_checkpoint == "Qwen/Qwen3.6-27B"
    assert len(manifest.tensors) == 1
    tensor = manifest.tensors[0]
    assert tensor.role == "linear_attn_beta"
    assert tensor.layer_index == 0
    assert tensor.qweight == "layer0.linear_attn_beta.qweight.u32.bin"
    assert tensor.calibration["method"] == "awq_activation_weighted"
    assert (target / tensor.qweight).stat().st_size == 640 * 48 * 4
    assert (target / tensor.scales).stat().st_size == 40 * 48 * 2
    assert (target / tensor.zeros).stat().st_size == 40 * 48 * 2

    matrix = load_packed_int4_matrix(tensor, target, device="cpu")
    assert matrix.role == "linear_attn_beta"
    assert matrix.qweight.shape == (640, 48)
    assert matrix.scales.shape == (40, 48)
    assert matrix.zeros.shape == (40, 48)


def test_import_qmi4_artifacts_rejects_wrong_role_shape(tmp_path: Path) -> None:
    source = tmp_path / "prototype"
    target = tmp_path / "metal_marlin"
    source.mkdir()
    _write_qmi4(source / "q.qmi4", out_features=48)

    with pytest.raises(ValueError, match="maps to linear_attn_q"):
        import_qmi4_artifacts(source, target)


def test_role_from_legacy_name_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported Qwen3.6-27B legacy tensor"):
        role_from_legacy_name("unknown.qmi4")
