"""Compatibility helpers for legacy Metal context imports."""

from __future__ import annotations

from .metal_dispatch import MetalKernelLibrary, get_default_library


def get_metal_kernel_library() -> MetalKernelLibrary:
    """Return the process-wide default Metal kernel library."""
    return get_default_library()
