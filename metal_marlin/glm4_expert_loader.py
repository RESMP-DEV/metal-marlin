"""Utilities for loading GLM-4 MoE expert MMFP4 weights from safetensors."""

from __future__ import annotations

import torch

from .mmfp4_loader import MMFP4ModelLoader


def _load_expert_projection(
    loader: MMFP4ModelLoader,
    layer_idx: int,
    expert_idx: int,
    proj_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one expert projection in row-packed format.

    Safetensors stores packed FP4 weights as ``[out_features, in_features//8]``.
    mmfp4_gemm expects row-packed ``[out_features, in_features//8]``.

    Scales are stored as ``[out_features, n_groups]`` which matches the
    Metal kernel's expected layout (B_scales[col * N_GROUPS + group_idx]).

    Returns:
        Tuple of (packed_weights, scales) both in [out, ...] layout.
    """
    tensor_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"
    qweight, scales = loader.get_quantized_weight(tensor_name)

    if qweight.ndim != 2:
        raise ValueError(
            f"Expected 2D qweight for {tensor_name}, got shape={tuple(qweight.shape)}"
        )
    if scales.ndim != 2:
        raise ValueError(
            f"Expected 2D scales for {tensor_name}, got shape={tuple(scales.shape)}"
        )

    # Keep row-packed layout [out, in//8] - do NOT transpose
    packed = qweight.contiguous().to(dtype=torch.uint32)
    # Keep scales in [out, n_groups] layout - matches Metal kernel expectation
    # Metal shader: B_scales[col * N_GROUPS + group_idx] requires [out, n_groups]
    scales = scales.contiguous().to(dtype=torch.float16)
    return packed, scales


def load_expert_weights(
    loader: MMFP4ModelLoader,
    layer_idx: int,
    num_experts: int = 64,
) -> dict[str, torch.Tensor]:
    """Load all expert weights for a single MoE layer.

    Returns a dictionary with stacked tensors:
    - ``gate_proj_packed``: ``[num_experts, out, in//8]`` uint32 (row-packed)
    - ``gate_proj_scales``: ``[num_experts, out, n_groups]`` fp16
    - ``up_proj_packed``: ``[num_experts, out, in//8]`` uint32 (row-packed)
    - ``up_proj_scales``: ``[num_experts, out, n_groups]`` fp16
    - ``down_proj_packed``: ``[num_experts, out, in//8]`` uint32 (row-packed)
    - ``down_proj_scales``: ``[num_experts, out, n_groups]`` fp16

    Notes:
    - Packed weights are kept in row-packed format [out, in//8] as stored
      in safetensors, which matches mmfp4_gemm's expected layout.
    - Scales are kept in [out, n_groups] format as stored in safetensors,
      which matches the Metal kernel's expected indexing pattern.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")

    packed_by_proj: dict[str, list[torch.Tensor]] = {
        "gate_proj": [],
        "up_proj": [],
        "down_proj": [],
    }
    scales_by_proj: dict[str, list[torch.Tensor]] = {
        "gate_proj": [],
        "up_proj": [],
        "down_proj": [],
    }

    for expert_idx in range(num_experts):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            packed, scales = _load_expert_projection(
                loader=loader,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                proj_name=proj_name,
            )
            packed_by_proj[proj_name].append(packed)
            scales_by_proj[proj_name].append(scales)

    return {
        "gate_proj_packed": torch.stack(packed_by_proj["gate_proj"], dim=0).contiguous(),
        "gate_proj_scales": torch.stack(scales_by_proj["gate_proj"], dim=0).contiguous(),
        "up_proj_packed": torch.stack(packed_by_proj["up_proj"], dim=0).contiguous(),
        "up_proj_scales": torch.stack(scales_by_proj["up_proj"], dim=0).contiguous(),
        "down_proj_packed": torch.stack(packed_by_proj["down_proj"], dim=0).contiguous(),
        "down_proj_scales": torch.stack(scales_by_proj["down_proj"], dim=0).contiguous(),
    }
