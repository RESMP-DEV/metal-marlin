"""Compatibility facade for the split kernel modules."""

from __future__ import annotations
import logging

from .. import kernels_core as _core
from ..kernels_core import *  # noqa: F401,F403

try:
    from .attention import build_attention_exports as _build_attention_exports
except ImportError:
    _build_attention_exports = None

try:
    from .moe import build_moe_exports as _build_moe_exports
except ImportError:
    _build_moe_exports = None



logger = logging.getLogger(__name__)

_ATTENTION_EXPORT_DEPENDENCIES = (
    "torch",
    "_private_buffer_from_tensor",
    "_params_buffer",
)

_MOE_EXPORT_DEPENDENCIES = (
    "np",
    "torch",
    "require_mps",
    "group_tokens_by_expert_full",
    "gather_for_experts",
    "scatter_expert_outputs",
    "get_default_library",
    "_ensure_kernel_compiled",
    "_private_buffer_from_tensor",
    "_params_buffer",
    "mps_tensor_to_metal_buffer",
    "dispatch_kernel",
    "get_shader_source",
    "marlin_gemm_fp4",
    "dequant_fp4",
    "FP4_PER_UINT",
)

if _build_attention_exports is not None and all(
    hasattr(_core, name) for name in _ATTENTION_EXPORT_DEPENDENCIES
):
    globals().update(
        _build_attention_exports(
            np=_core.np,
            torch=_core.torch,
            require_mps=_core.require_mps,
            get_default_library=_core.get_default_library,
            _ensure_kernel_compiled=_core._ensure_kernel_compiled,
            _private_buffer_from_tensor=_core._private_buffer_from_tensor,
            _params_buffer=_core._params_buffer,
            mps_tensor_to_metal_buffer=_core.mps_tensor_to_metal_buffer,
            dispatch_kernel=_core.dispatch_kernel,
            FP4_PER_UINT=_core.FP4_PER_UINT,
        )
    )

if _build_moe_exports is not None and all(
    hasattr(_core, name) for name in _MOE_EXPORT_DEPENDENCIES
):
    globals().update(
        _build_moe_exports(
            np=_core.np,
            torch=_core.torch,
            require_mps=_core.require_mps,
            group_tokens_by_expert_full=_core.group_tokens_by_expert_full,
            gather_for_experts=_core.gather_for_experts,
            scatter_expert_outputs=_core.scatter_expert_outputs,
            get_default_library=_core.get_default_library,
            _ensure_kernel_compiled=_core._ensure_kernel_compiled,
            _private_buffer_from_tensor=_core._private_buffer_from_tensor,
            _params_buffer=_core._params_buffer,
            mps_tensor_to_metal_buffer=_core.mps_tensor_to_metal_buffer,
            dispatch_kernel=_core.dispatch_kernel,
            get_shader_source=_core.get_shader_source,
            marlin_gemm_fp4=_core.marlin_gemm_fp4,
            dequant_fp4=_core.dequant_fp4,
            FP4_PER_UINT=_core.FP4_PER_UINT,
        )
    )

_cpp_ext = getattr(_core, "_cpp_ext", None)
__all__ = [name for name in globals() if not name.startswith("_")]
