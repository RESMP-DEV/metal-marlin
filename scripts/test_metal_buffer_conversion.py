#!/usr/bin/env python3
"""Minimal test for PyObjC Metal buffer creation paths.

Run:
  uv run python scripts/test_metal_buffer_conversion.py
"""

from __future__ import annotations

import ctypes
import logging

import Metal
import numpy as np
import torch



logger = logging.getLogger(__name__)

def _try(label: str, fn) -> None:
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: FAIL -> {type(exc).__name__}: {exc}")
        return
    print(f"{label}: OK -> {result}")


def main() -> None:
    logger.info("main starting")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device available")

    print("Metal device:", device)

    # 1) No-copy from a Python buffer (bytearray) - expected to work.
    backing = bytearray(1024)

    def _no_copy_bytearray() -> int:
        logger.debug("_no_copy_bytearray called")
        buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            backing, len(backing), Metal.MTLResourceStorageModeShared, None
        )
        return int(buf.length())

    _try("no-copy bytearray", _no_copy_bytearray)

    # 2) Copy from bytes - expected to work.
    arr = np.arange(16, dtype=np.float32)

    def _copy_bytes() -> int:
        logger.debug("_copy_bytes called")
        buf = device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
        )
        return int(buf.length())

    _try("copy bytes", _copy_bytes)

    # 3) No-copy from integer pointer - expected to fail with PyObjC conversion error.
    def _no_copy_int_ptr() -> int:
        logger.debug("_no_copy_int_ptr called")
        ptr = int(arr.ctypes.data)
        buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ptr, arr.nbytes, Metal.MTLResourceStorageModeShared, None
        )
        return int(buf.length())

    _try("no-copy int pointer", _no_copy_int_ptr)

    # 4) No-copy from ctypes array memoryview - may or may not work depending on PyObjC.
    def _no_copy_ctypes_mv() -> int:
        logger.debug("_no_copy_ctypes_mv called")
        c_arr = (ctypes.c_uint8 * arr.nbytes).from_address(int(arr.ctypes.data))
        mv = memoryview(c_arr)
        buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            mv, arr.nbytes, Metal.MTLResourceStorageModeShared, None
        )
        return int(buf.length())

    _try("no-copy ctypes memoryview", _no_copy_ctypes_mv)

    # 5) Torch MPS path (if available) - likely to reproduce the failure.
    if torch.backends.mps.is_available():
        t = torch.arange(16, device="mps", dtype=torch.float32)

        def _no_copy_mps_ptr() -> int:
            logger.debug("_no_copy_mps_ptr called")
            storage = t.untyped_storage()
            ptr = storage.data_ptr()
            size = storage.nbytes()
            buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                ptr, size, Metal.MTLResourceStorageModeShared, None
            )
            return int(buf.length())

        _try("no-copy torch mps ptr", _no_copy_mps_ptr)

        def _copy_mps_cpu() -> int:
            logger.debug("_copy_mps_cpu called")
            cpu_arr = t.cpu().numpy()
            buf = device.newBufferWithBytes_length_options_(
                cpu_arr.tobytes(), cpu_arr.nbytes, Metal.MTLResourceStorageModeShared
            )
            return int(buf.length())

        _try("copy torch mps via cpu", _copy_mps_cpu)
    else:
        print("torch MPS not available; skipping MPS tests")


if __name__ == "__main__":
    main()
