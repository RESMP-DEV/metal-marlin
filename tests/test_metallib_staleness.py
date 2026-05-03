"""Tests for metallib staleness detection."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import metal_marlin.metallib_loader as metallib_loader


def _write_shader(path: Path, contents: str) -> Path:
    logger.info("_write_shader called with path=%s, contents=%s", path, contents)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def _make_fake_metallib_tree(tmp_path: Path) -> tuple[Path, Path]:
    logger.debug("_make_fake_metallib_tree called with tmp_path=%s", tmp_path)
    metallib_path = tmp_path / "metal_marlin" / "lib" / "metal_marlin.metallib"
    metallib_path.parent.mkdir(parents=True, exist_ok=True)
    metallib_path.write_bytes(b"fake metallib")

    shader_path = _write_shader(
        tmp_path / "src" / "base_kernel.metal",
        "kernel void base_kernel() {}\n",
    )
    return metallib_path, shader_path


def test_staleness_details_detect_modified_added_and_removed_files(tmp_path: Path) -> None:
    logger.info("running test_staleness_details_detect_modified_added_and_removed_files")
    metallib_path, shader_path = _make_fake_metallib_tree(tmp_path)

    metallib_loader.save_checksum_manifest(metallib_path)
    clean = metallib_loader.get_staleness_details(metallib_path)
    assert clean["has_manifest"] is True
    assert clean["is_stale"] is False
    assert clean["reason"] == "checksums match"

    shader_path.write_text("kernel void base_kernel() { threadgroup_barrier(0); }\n")
    modified = metallib_loader.get_staleness_details(metallib_path)
    assert modified["is_stale"] is True
    assert modified["reason"] == "1 modified"
    assert modified["modified_files"] == ["src/base_kernel.metal"]
    assert metallib_loader.is_metallib_stale(metallib_path) is True

    shader_path.write_text("kernel void base_kernel() {}\n")
    metallib_loader.save_checksum_manifest(metallib_path)

    added_shader = _write_shader(
        tmp_path / "src" / "added_kernel.metal",
        "kernel void added_kernel() {}\n",
    )
    added = metallib_loader.get_staleness_details(metallib_path)
    assert added["is_stale"] is True
    assert added["reason"] == "1 added"
    assert added["added_files"] == ["src/added_kernel.metal"]

    metallib_loader.save_checksum_manifest(metallib_path)
    added_shader.unlink()
    removed = metallib_loader.get_staleness_details(metallib_path)
    assert removed["is_stale"] is True
    assert removed["reason"] == "1 removed"
    assert removed["removed_files"] == ["src/added_kernel.metal"]


def test_staleness_details_fall_back_to_source_hash(tmp_path: Path) -> None:
    logger.info("running test_staleness_details_fall_back_to_source_hash")
    metallib_path, shader_path = _make_fake_metallib_tree(tmp_path)

    metallib_loader.save_source_hash(metallib_path)
    clean = metallib_loader.get_staleness_details(metallib_path)
    assert clean["has_manifest"] is False
    assert clean["is_stale"] is False
    assert clean["reason"] == "source hash matches"

    shader_path.write_text("kernel void base_kernel() { /* changed */ }\n")
    stale = metallib_loader.get_staleness_details(metallib_path)
    assert stale["is_stale"] is True
    assert stale["reason"] == "aggregate source hash mismatch"
    assert metallib_loader.is_metallib_stale(metallib_path) is True


def test_load_metallib_warns_for_stale_cached_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger.info("running test_load_metallib_warns_for_stale_cached_library")
    metallib_path, shader_path = _make_fake_metallib_tree(tmp_path)
    metallib_loader.save_checksum_manifest(metallib_path)

    cached_library = object()
    monkeypatch.setattr(metallib_loader, "require_metal", lambda: None)
    monkeypatch.setattr(metallib_loader, "_cached_library", cached_library)
    monkeypatch.setattr(metallib_loader, "_cached_path", metallib_path)

    shader_path.write_text("kernel void base_kernel() { /* stale */ }\n")

    with caplog.at_level(logging.WARNING):
        result = metallib_loader.load_metallib(metallib_path)

    assert result is cached_library
    assert "Metal shaders modified since last metallib build [1 modified]." in caplog.text
    assert "src/base_kernel.metal" in caplog.text
