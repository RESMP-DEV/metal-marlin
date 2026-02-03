"""Tests for the Metal Marlin C++ extension (_cpp_ext)."""

from __future__ import annotations

import ctypes
import gc

import pytest

from metal_marlin._compat import HAS_CPP_EXT

try:
    import metal_marlin._cpp_ext as cpp_ext
    _CPP_EXT_AVAILABLE = True
except Exception:
    cpp_ext = None
    _CPP_EXT_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _CPP_EXT_AVAILABLE, reason="C++ extension not available")


def _require_context():
    if cpp_ext is None:
        pytest.skip("C++ extension not available")
    try:
        return cpp_ext.MetalContext()
    except Exception as exc:
        pytest.skip(f"MetalContext unavailable: {exc}")


def test_extension_flag_matches_import():
    assert HAS_CPP_EXT == _CPP_EXT_AVAILABLE


def test_module_exports():
    assert cpp_ext is not None
    assert hasattr(cpp_ext, "MetalContext")
    assert hasattr(cpp_ext, "BufferPool")
    assert hasattr(cpp_ext, "ManagedBuffer")
    assert hasattr(cpp_ext, "BatchDispatch")
    assert hasattr(cpp_ext, "dispatch_kernel")
    assert hasattr(cpp_ext, "create_buffer")
    assert hasattr(cpp_ext, "create_buffer_from_bytes")
    assert hasattr(cpp_ext, "create_buffer_from_ptr")
    assert hasattr(cpp_ext, "align_buffer_size")

    assert isinstance(cpp_ext.CACHE_LINE_SIZE, int)
    assert isinstance(cpp_ext.PAGE_SIZE, int)
    assert isinstance(cpp_ext.LARGE_BUFFER_THRESHOLD, int)
    assert cpp_ext.CACHE_LINE_SIZE > 0
    assert cpp_ext.PAGE_SIZE > 0
    assert cpp_ext.LARGE_BUFFER_THRESHOLD > 0


def test_align_buffer_size():
    cache = cpp_ext.CACHE_LINE_SIZE
    page = cpp_ext.PAGE_SIZE
    threshold = cpp_ext.LARGE_BUFFER_THRESHOLD

    small = cache - 1
    expected_small = (small + cache - 1) & ~(cache - 1)
    assert cpp_ext.align_buffer_size(small) == expected_small

    large = threshold + 1
    expected_large = (large + page - 1) & ~(page - 1)
    assert cpp_ext.align_buffer_size(large) == expected_large


def test_metal_context_properties():
    ctx = _require_context()

    name = ctx.device_name()
    family = ctx.gpu_family()

    assert isinstance(name, str)
    assert name
    assert isinstance(family, int)
    assert family >= 7


def test_buffer_pool_reuse():
    ctx = _require_context()
    pool = ctx.buffer_pool

    size = 1024
    hits_before = pool.hits()
    misses_before = pool.misses()

    buf = cpp_ext.create_buffer(ctx, size, True)
    assert buf.length() >= size
    assert buf.data_ptr() != 0

    del buf
    gc.collect()

    pooled_after_release = pool.pooled_count()
    assert pooled_after_release >= 1

    buf2 = cpp_ext.create_buffer(ctx, size, True)
    del buf2
    gc.collect()

    assert pool.misses() >= misses_before + 1
    assert pool.hits() >= hits_before + 1


def test_create_buffer_from_bytes_roundtrip():
    ctx = _require_context()

    payload = b"metal-marlin-cpp-ext"
    buf = cpp_ext.create_buffer_from_bytes(ctx, payload, False)

    assert buf.length() >= len(payload)
    ptr = buf.data_ptr()
    assert ptr != 0

    read_back = ctypes.string_at(ptr, len(payload))
    assert read_back == payload

    del buf
    gc.collect()


def test_create_buffer_from_ptr():
    ctx = _require_context()

    payload = b"direct-pointer"
    raw = ctypes.create_string_buffer(payload)
    ptr = ctypes.addressof(raw)

    buf = cpp_ext.create_buffer_from_ptr(ctx, ptr, len(payload))

    assert buf.length() >= len(payload)
    assert buf.data_ptr() != 0

    del buf
    gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
