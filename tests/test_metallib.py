"""Tests for precompiled metallib loading."""
from pathlib import Path

import pytest


@pytest.fixture
def metallib_path():
    return Path(__file__).parent.parent / "metal_marlin" / "lib" / "metal_marlin.metallib"


def test_metallib_exists(metallib_path):
    """Metallib file should exist after build."""
    assert metallib_path.exists(), "Run ./scripts/build_metallib.sh first"


def test_load_metallib():
    from metal_marlin.metallib_loader import load_metallib
    lib = load_metallib()
    assert lib is not None


def test_get_precompiled_library_cached():
    from metal_marlin.metallib_loader import clear_cache, get_precompiled_library
    clear_cache()
    lib1 = get_precompiled_library()
    lib2 = get_precompiled_library()
    assert lib1 is lib2, "Should return cached instance"


def test_kernel_lookup():
    from metal_marlin.metallib_loader import get_kernel_from_metallib
    # Use actual kernel name from metallib (not "layernorm_forward")
    kernel = get_kernel_from_metallib("layernorm")
    assert kernel is not None


def test_missing_kernel_returns_none():
    from metal_marlin.metallib_loader import get_kernel_from_metallib
    result = get_kernel_from_metallib("nonexistent_kernel_xyz")
    assert result is None


def test_version_info():
    from metal_marlin.metallib_loader import get_metallib_version
    version = get_metallib_version()
    assert "build_date" in version or "error" in version
