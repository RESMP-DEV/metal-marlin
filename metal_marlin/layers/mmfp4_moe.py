"""MMFP4 Mixture-of-Experts layer with grouped batched dispatch.

Implements batched expert execution for GLM-4.7-Flash (64 experts, top-4).
Supports MMFP4-quantized expert weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmfp4_linear import MMFP4Linear

_first_call = True

if TYPE_CHECKING:
    pass

# Lazy import to avoid circular dependency issues
_moe_dispatch_module: Any = None


def _get_dispatch_module() -> Any:
    global _moe_dispatch_module
    if _moe_dispatch_module is None:
        from .. import moe_dispatch as md
        _moe_dispatch_module = md
    return _moe_dispatch_module


def _get_config_value(config: Any, names: Sequence[str], default: int) -> int:
    """Read the first available integer-like value from config."""
    if isinstance(config, dict):
        for name in names:
            value = config.get(name)
            if value is not None:
                return int(value)
        return int(default)

    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return int(value)
    return int(default)


def _copy_linear_weight(linear: nn.Linear, tensor: torch.Tensor, key: str) -> None:
    """Copy a weight tensor into a linear layer, auto-handling transposed layout."""
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor for {key}, got shape={tuple(tensor.shape)}")

    expected = linear.weight.shape
    if tensor.shape == expected:
        src = tensor
    elif tensor.T.shape == expected:
        src = tensor.T
    else:
        raise ValueError(
            f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {tuple(expected)}"
        )

    linear.weight.data.copy_(
        src.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _copy_mmfp4_weight(
    mmfp4_linear: MMFP4Linear,
    packed_tensor: torch.Tensor,
    scales_tensor: torch.Tensor,
    key: str,
) -> None:
    """Copy MMFP4 packed weights and scales into an MMFP4Linear layer."""
    # Handle packed weights
    expected_packed = mmfp4_linear.packed_weights.shape
    if packed_tensor.shape == expected_packed:
        src_packed = packed_tensor
    elif packed_tensor.T.shape == expected_packed:
        src_packed = packed_tensor.T
    else:
        raise ValueError(
            f"Shape mismatch for {key}.weight: got {tuple(packed_tensor.shape)}, "
            f"expected {tuple(expected_packed)}"
        )
    mmfp4_linear.packed_weights.data.copy_(
        src_packed.to(device=mmfp4_linear.packed_weights.device,
                      dtype=torch.uint32)
    )

    # Handle scales - they may be transposed [out, n_groups] vs [n_groups, out]
    expected_scales = mmfp4_linear.scales.shape
    if scales_tensor.shape == expected_scales:
        src_scales = scales_tensor
    elif scales_tensor.T.shape == expected_scales:
        src_scales = scales_tensor.T
    else:
        raise ValueError(
            f"Shape mismatch for {key}.scales: got {tuple(scales_tensor.shape)}, "
            f"expected {tuple(expected_scales)}"
        )
    mmfp4_linear.scales.data.copy_(
        src_scales.to(device=mmfp4_linear.scales.device,
                      dtype=mmfp4_linear.scales.dtype)
    )


class MMFP4Expert(nn.Module):
    """Single SwiGLU expert using MMFP4-quantized linear layers."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.group_size = group_size

        # Create placeholder MMFP4Linear layers - weights loaded later
        self.gate_proj = _make_placeholder_mmfp4_linear(
            hidden_size, moe_intermediate_size, group_size
        )
        self.up_proj = _make_placeholder_mmfp4_linear(
            hidden_size, moe_intermediate_size, group_size
        )
        self.down_proj = _make_placeholder_mmfp4_linear(
            moe_intermediate_size, hidden_size, group_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def _make_placeholder_mmfp4_linear(
    in_features: int,
    out_features: int,
    group_size: int,
) -> MMFP4Linear:
    """Create MMFP4Linear with placeholder (zeros) weights."""
    in_features_aligned = ((in_features + 7) // 8) * 8
    n_groups = (in_features + group_size - 1) // group_size
    packed_weights = torch.zeros(
        (out_features, in_features_aligned // 8),
        dtype=torch.uint32,
    )
    scales = torch.ones((n_groups, out_features), dtype=torch.float16)
    return MMFP4Linear(
        packed_weights=packed_weights,
        scales=scales,
        bias=None,
        group_size=group_size,
    )


class MMFP4MoE(nn.Module):
    """MMFP4 MoE layer with top-k routing and grouped batched expert execution."""

    def __init__(
        self,
        n_experts: int = 64,
        n_experts_per_tok: int = 4,
        hidden_size: int = 2048,
        moe_intermediate_size: int = 1536,
        group_size: int = 128,
        has_shared_expert: bool = True,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be > 0")
        if n_experts_per_tok <= 0:
            raise ValueError("n_experts_per_tok must be > 0")
        if n_experts_per_tok > n_experts:
            raise ValueError("n_experts_per_tok must be <= n_experts")

        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.group_size = group_size
        self.has_shared_expert = has_shared_expert

        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MMFP4Expert(hidden_size, moe_intermediate_size, group_size)
                for _ in range(n_experts)
            ]
        )

        # Shared expert that's always active (GLM-4 architecture)
        if has_shared_expert:
            self.shared_experts = MMFP4Expert(
                hidden_size, moe_intermediate_size, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run top-k routing + grouped batched expert execution + weighted combine."""
        global _first_call

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected input hidden_size={self.hidden_size}, got {x.shape[-1]}"
            )

        dispatch = _get_dispatch_module()
        original_shape = x.shape
        hidden_flat = x.reshape(-1, self.hidden_size)
        if hidden_flat.shape[0] == 0:
            return x

        # Router in router dtype for numerical consistency.
        gate_input = hidden_flat.to(self.gate.weight.dtype)
        gate_logits = self.gate(gate_input)

        topk_logits, topk_indices = torch.topk(
            gate_logits,
            k=self.n_experts_per_tok,
            dim=-1,
            largest=True,
            sorted=False,
        )
        topk_weights = F.softmax(
            topk_logits, dim=-1, dtype=torch.float32).to(hidden_flat.dtype)

        # Group assignments by expert, run each expert once on its contiguous slice.
        dispatch_info = dispatch.group_tokens_by_expert_full(
            topk_indices, self.n_experts)
        expert_inputs = dispatch.gather_for_experts(hidden_flat, dispatch_info)
        expert_outputs = hidden_flat.new_empty(
            (expert_inputs.shape[0], self.hidden_size))

        # Try fused dispatch first, fall back to sequential if unavailable
        used_fused = False
        try:
            expert_outputs = dispatch.dispatch_mmfp4_experts_fused(
                expert_inputs, self.experts, dispatch_info, self.n_experts
            )
            used_fused = True
        except (ValueError, NotImplementedError) as e:
            # Fused dispatch unavailable (non-MPS device or kernel not built)
            # Fall back to sequential expert dispatch
            expert_offsets_cpu = dispatch_info.expert_offsets.cpu()
            for expert_idx in range(self.n_experts):
                start = int(expert_offsets_cpu[expert_idx].item())
                end = int(expert_offsets_cpu[expert_idx + 1].item())
                if start == end:
                    continue
                expert = self.experts[expert_idx]
                chunk = expert_inputs[start:end].to(torch.float16)
                chunk_out = expert(chunk)
                expert_outputs[start:end] = chunk_out.to(expert_outputs.dtype)

        if _first_call:
            _first_call = False

        # Keep expert execution asynchronous; synchronize once per layer.
        if hidden_flat.is_mps:
            torch.mps.synchronize()

        combined = dispatch.scatter_expert_outputs(
            expert_outputs, topk_weights, dispatch_info)

        # Add shared expert output (always active)
        if self.has_shared_expert and hasattr(self, 'shared_experts'):
            shared_input = hidden_flat.to(torch.float16)
            shared_output = self.shared_experts(shared_input)
            combined = combined + shared_output.to(combined.dtype)

        return combined.reshape(original_shape)

    @staticmethod
    def _layer_key_matches(key: str, layer_idx: int) -> bool:
        tokens = (
            f".layers.{layer_idx}.",
            f"layers.{layer_idx}.",
            f".h.{layer_idx}.",
            f"h.{layer_idx}.",
        )
        return any(token in key for token in tokens)

    @classmethod
    def _find_tensor(
        cls,
        tensors: dict[str, torch.Tensor],
        *,
        keys: Sequence[str],
        suffixes: Sequence[str],
        layer_idx: int,
    ) -> tuple[str, torch.Tensor] | None:
        for key in keys:
            if key in tensors:
                return key, tensors[key]

        for key in sorted(tensors.keys()):
            if not cls._layer_key_matches(key, layer_idx):
                continue
            if any(key.endswith(suffix) for suffix in suffixes):
                return key, tensors[key]
        return None

    @classmethod
    def _split_gate_up(
        cls,
        gate_up: torch.Tensor,
        hidden_size: int,
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gate_up.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor for {key}, got shape={tuple(gate_up.shape)}")

        # Standard linear layout: [2*intermediate, hidden]
        if gate_up.shape[1] == hidden_size and gate_up.shape[0] % 2 == 0:
            half = gate_up.shape[0] // 2
            return gate_up[:half], gate_up[half:]

        # Transposed fallback: [hidden, 2*intermediate]
        if gate_up.shape[0] == hidden_size and gate_up.shape[1] % 2 == 0:
            half = gate_up.shape[1] // 2
            return gate_up[:, :half], gate_up[:, half:]

        raise ValueError(
            f"Cannot split fused gate_up tensor {key} with shape={tuple(gate_up.shape)} "
            f"for hidden_size={hidden_size}"
        )

    @classmethod
    def from_hf_weights(cls, layer_idx: int, tensors: dict, config) -> MMFP4MoE:
        """Load MoE router + experts from a HuggingFace tensor dictionary."""
        layer_prefixes = (
            f"model.layers.{layer_idx}.mlp",
            f"model.layers.{layer_idx}.block_sparse_moe",
            f"model.layers.{layer_idx}.moe",
            f"transformer.layers.{layer_idx}.mlp",
            f"transformer.layers.{layer_idx}.block_sparse_moe",
            f"layers.{layer_idx}.mlp",
        )

        gate_keys = []
        for prefix in layer_prefixes:
            gate_keys.extend(
                (
                    f"{prefix}.gate.weight",
                    f"{prefix}.router.weight",
                    f"{prefix}.moe_gate.weight",
                    f"{prefix}.expert_gate.weight",
                )
            )

        gate_match = cls._find_tensor(
            tensors,
            keys=gate_keys,
            suffixes=(
                ".mlp.gate.weight",
                ".mlp.router.weight",
                ".block_sparse_moe.gate.weight",
                ".moe.gate.weight",
                ".moe_gate.weight",
                ".expert_gate.weight",
                ".router.weight",
            ),
            layer_idx=layer_idx,
        )
        if gate_match is None:
            raise KeyError(
                f"Could not find router/gate weight for MoE layer {layer_idx}")

        gate_key, gate_weight = gate_match
        if gate_weight.ndim != 2:
            raise ValueError(
                f"Expected router weight to be 2D for {gate_key}, got {tuple(gate_weight.shape)}"
            )

        inferred_n_experts = int(gate_weight.shape[0])
        inferred_hidden = int(gate_weight.shape[1])

        n_experts = _get_config_value(
            config,
            ("num_local_experts", "n_routed_experts", "num_experts"),
            inferred_n_experts,
        )
        n_experts = inferred_n_experts if n_experts != inferred_n_experts else n_experts

        hidden_size = _get_config_value(
            config, ("hidden_size", "d_model"), inferred_hidden)
        hidden_size = inferred_hidden if hidden_size != inferred_hidden else hidden_size

        n_experts_per_tok = _get_config_value(
            config,
            ("num_experts_per_tok", "num_selected_experts", "num_experts_per_token"),
            4,
        )
        group_size = _get_config_value(config, ("group_size",), 128)

        moe_intermediate_size = _get_config_value(
            config,
            ("moe_intermediate_size", "ffn_hidden_size", "intermediate_size"),
            1536,
        )

        # Attempt to infer per-expert hidden size directly from expert 0 if available.
        expert0_patterns = (
            ".experts.0.gate_proj.weight",
            ".experts.0.w1.weight",
            ".experts.0.gate_up_proj.weight",
        )
        expert0 = cls._find_tensor(
            tensors,
            keys=(),
            suffixes=expert0_patterns,
            layer_idx=layer_idx,
        )
        if expert0 is not None:
            expert0_key, expert0_weight = expert0
            if expert0_weight.ndim == 2:
                if expert0_weight.shape[1] == hidden_size:
                    out_dim = int(expert0_weight.shape[0])
                elif expert0_weight.shape[0] == hidden_size:
                    out_dim = int(expert0_weight.shape[1])
                else:
                    out_dim = moe_intermediate_size
                if expert0_key.endswith(".gate_up_proj.weight"):
                    if out_dim % 2 != 0:
                        raise ValueError(
                            f"Expected even fused gate_up dimension for {expert0_key}, got {out_dim}"
                        )
                    moe_intermediate_size = out_dim // 2
                else:
                    moe_intermediate_size = out_dim

        model = cls(
            n_experts=n_experts,
            n_experts_per_tok=n_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            group_size=group_size,
        )

        _copy_linear_weight(model.gate, gate_weight, gate_key)

        for expert_idx in range(n_experts):
            expert_roots = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}",
                f"model.layers.{layer_idx}.mlp.block_sparse_moe.experts.{expert_idx}",
                f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}",
                f"model.layers.{layer_idx}.moe.experts.{expert_idx}",
                f"transformer.layers.{layer_idx}.mlp.experts.{expert_idx}",
                f"transformer.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}",
                f"layers.{layer_idx}.mlp.experts.{expert_idx}",
            )

            gate_tensor: torch.Tensor | None = None
            up_tensor: torch.Tensor | None = None
            down_tensor: torch.Tensor | None = None
            gate_name = up_name = down_name = ""

            for root in expert_roots:
                g = f"{root}.gate_proj.weight"
                u = f"{root}.up_proj.weight"
                d = f"{root}.down_proj.weight"
                if g in tensors and u in tensors and d in tensors:
                    gate_tensor, up_tensor, down_tensor = tensors[g], tensors[u], tensors[d]
                    gate_name, up_name, down_name = g, u, d
                    break

                w1 = f"{root}.w1.weight"
                w2 = f"{root}.w2.weight"
                w3 = f"{root}.w3.weight"
                if w1 in tensors and w2 in tensors and w3 in tensors:
                    # Mixtral naming: w1=gate, w3=up, w2=down
                    gate_tensor, up_tensor, down_tensor = tensors[w1], tensors[w3], tensors[w2]
                    gate_name, up_name, down_name = w1, w3, w2
                    break

                gate_up = f"{root}.gate_up_proj.weight"
                d2 = f"{root}.down_proj.weight"
                if gate_up in tensors and d2 in tensors:
                    gate_tensor, up_tensor = cls._split_gate_up(
                        tensors[gate_up], hidden_size, gate_up
                    )
                    down_tensor = tensors[d2]
                    gate_name, up_name, down_name = (
                        f"{gate_up}[:half]",
                        f"{gate_up}[half:]",
                        d2,
                    )
                    break

            if gate_tensor is None or up_tensor is None or down_tensor is None:
                gate_match = cls._find_tensor(
                    tensors,
                    keys=(),
                    suffixes=(
                        f".experts.{expert_idx}.gate_proj.weight",
                        f".experts.{expert_idx}.w1.weight",
                    ),
                    layer_idx=layer_idx,
                )
                up_match = cls._find_tensor(
                    tensors,
                    keys=(),
                    suffixes=(
                        f".experts.{expert_idx}.up_proj.weight",
                        f".experts.{expert_idx}.w3.weight",
                    ),
                    layer_idx=layer_idx,
                )
                down_match = cls._find_tensor(
                    tensors,
                    keys=(),
                    suffixes=(
                        f".experts.{expert_idx}.down_proj.weight",
                        f".experts.{expert_idx}.w2.weight",
                    ),
                    layer_idx=layer_idx,
                )

                if gate_match is not None and up_match is not None and down_match is not None:
                    gate_name, gate_tensor = gate_match
                    up_name, up_tensor = up_match
                    down_name, down_tensor = down_match
                else:
                    gate_up_match = cls._find_tensor(
                        tensors,
                        keys=(),
                        suffixes=(
                            f".experts.{expert_idx}.gate_up_proj.weight",),
                        layer_idx=layer_idx,
                    )
                    if gate_up_match is None or down_match is None:
                        raise KeyError(
                            "Missing expert tensors for layer "
                            f"{layer_idx}, expert {expert_idx}"
                        )
                    gate_up_name, gate_up_tensor = gate_up_match
                    down_name, down_tensor = down_match
                    gate_tensor, up_tensor = cls._split_gate_up(
                        gate_up_tensor, hidden_size, gate_up_name
                    )
                    gate_name = f"{gate_up_name}[:half]"
                    up_name = f"{gate_up_name}[half:]"

            expert = model.experts[expert_idx]
            _copy_linear_weight(expert.gate_proj, gate_tensor, gate_name)
            _copy_linear_weight(expert.up_proj, up_tensor, up_name)
            _copy_linear_weight(expert.down_proj, down_tensor, down_name)

        return model


__all__ = ["MMFP4Expert", "MMFP4MoE"]
