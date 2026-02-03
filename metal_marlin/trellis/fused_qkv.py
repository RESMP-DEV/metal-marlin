"""Standard QKV Attention with trellis-quantized projections.

Provides fused Q/K/V projection using a single kernel launch instead of
3 separate TrellisLinear calls.

Usage:
    from metal_marlin.trellis.fused_qkv import FusedQKVLinear

    q_proj = TrellisLinear(...)
    k_proj = TrellisLinear(...)
    v_proj = TrellisLinear(...)

    fused = FusedQKVLinear(q_proj, k_proj, v_proj)
    q, k, v = fused(hidden_states)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..metal_dispatch import HAS_METAL, HAS_MPS, MetalKernelLibrary
from .dispatch import dispatch_fused_qkv_trellis

if TYPE_CHECKING:
    from .linear import TrellisLinear


class FusedQKVLinear(nn.Module):
    """Fused Q/K/V projection using single kernel launch.

    Instead of:
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)

    Computes:
        q, k, v = fused_qkv_kernel(x, q_proj, k_proj, v_proj)

    Saves:
        - 2 kernel launches per attention layer
        - 2x read of input activations (x loaded once, reused for Q/K/V)

    For a 47-layer model: 94 fewer kernel launches per forward pass.
    """

    def __init__(
        self,
        q_proj: TrellisLinear,
        k_proj: TrellisLinear,
        v_proj: TrellisLinear,
    ) -> None:
        """Initialize FusedQKVLinear.

        Args:
            q_proj: TrellisLinear for Q projection
            k_proj: TrellisLinear for K projection
            v_proj: TrellisLinear for V projection
        """
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        if not (HAS_METAL and HAS_MPS):
            raise RuntimeError(
                "FusedQKVLinear requires Metal and MPS. "
                "Install with: pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
            )

        self._lib: MetalKernelLibrary | None = None

    def _get_lib(self) -> MetalKernelLibrary:
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with fused QKV projection.

        Args:
            x: Input tensor [..., in_features] where in_features is common
               to all three projections.

        Returns:
            Tuple of (q, k, v) output tensors:
                - q: [..., q_proj.out_features]
                - k: [..., k_proj.out_features]
                - v: [..., v_proj.out_features]
        """
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.q_proj.in_features)

        if x.dtype != torch.float16:
            x = x.to(torch.float16)
            x_flat = x_flat.to(torch.float16)

        lib = self._get_lib()

        q, k, v = dispatch_fused_qkv_trellis(
            lib,
            x_flat,
            self.q_proj,
            self.k_proj,
            self.v_proj,
        )

        q = q.view(*batch_shape, self.q_proj.out_features)
        k = k.view(*batch_shape, self.k_proj.out_features)
        v = v.view(*batch_shape, self.v_proj.out_features)

        return q, k, v

    def extra_repr(self) -> str:
        return (
            f"Q: {self.q_proj.in_features}->{self.q_proj.out_features}, "
            f"K: {self.k_proj.in_features}->{self.k_proj.out_features}, "
            f"V: {self.v_proj.in_features}->{self.v_proj.out_features}"
        )
