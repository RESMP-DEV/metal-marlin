"""Backward-compatible MMFP4 MoE exports.

Decode optimization with `_decode_fast_path` and `cached_expert` routing state
is implemented on `MMFP4MoE` in `metal_marlin.layers.mmfp4_moe`.
"""

from __future__ import annotations

from .layers.mmfp4_moe import MMFP4Expert, MMFP4MoE


class MMFP4MoEExperts(MMFP4MoE):
    """Compatibility alias for older MMFP4 MoE expert naming."""


__all__ = ["MMFP4Expert", "MMFP4MoE", "MMFP4MoEExperts"]
