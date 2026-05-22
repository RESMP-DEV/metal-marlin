from __future__ import annotations

import os

import pytest

from metal_marlin.qwen36_27b_validation import (
    default_block_layer0_manifest_path,
    default_layer0_manifest_path,
    validate_layer0_mlp_artifacts,
    validate_layer0_qkvb_artifacts,
)


def test_optional_layer0_qmi4_artifacts_match_fused_projection() -> None:
    if os.environ.get("METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS") != "1":
        pytest.skip("set METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS=1")
    manifest = default_layer0_manifest_path()
    if not manifest.exists():
        pytest.skip(f"local Qwen3.6-27B layer-0 artifact manifest missing: {manifest}")

    result = validate_layer0_qkvb_artifacts(manifest)

    result.assert_within(atol=0.08, rtol=0.02)


def test_optional_layer0_block_qkvb_artifacts_match_fused_projection() -> None:
    if os.environ.get("METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS") != "1":
        pytest.skip("set METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS=1")
    manifest = default_block_layer0_manifest_path()
    if not manifest.exists():
        pytest.skip(
            f"local Qwen3.6-27B layer-0 block artifact manifest missing: {manifest}"
        )

    result = validate_layer0_qkvb_artifacts(manifest)

    result.assert_within(atol=0.08, rtol=0.02)


def test_optional_layer0_block_mlp_artifacts_match_fused_kernels() -> None:
    if os.environ.get("METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS") != "1":
        pytest.skip("set METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS=1")
    manifest = default_block_layer0_manifest_path()
    if not manifest.exists():
        pytest.skip(
            f"local Qwen3.6-27B layer-0 block artifact manifest missing: {manifest}"
        )

    result = validate_layer0_mlp_artifacts(manifest)

    result.assert_within(atol=0.08, rtol=0.02)
