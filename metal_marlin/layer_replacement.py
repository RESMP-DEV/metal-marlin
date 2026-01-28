"""Utilities for replacing nn.Linear layers with MetalQuantizedLinear."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn

from .inference_metal import MetalQuantizedLinear
from .mr_gptq import MRGPTQQuantizer, QuantizationFormat

_SUPPORTED_FORMATS = {"fp4", "int4", "nf4"}


class MoEExperts(Protocol):
    """Protocol for MoE expert containers with fused expert weights."""

    gate_up_proj: Any
    down_proj: Any


class MoEModule(Protocol):
    """Protocol for MoE modules that expose fused expert weights."""

    experts: MoEExperts


class MetalQuantizedMoE(nn.Module):
    """Container for quantized MoE expert weights (routed + optional shared).

    Holds packed FP4/INT4/NF4 expert weights and per-group scales in the
    layout expected by moe_expert_gemm.metal:
        gate_up_weight_packed: [num_experts, K/8, N]
        gate_up_scales:       [num_experts, K/group_size, N]
        down_weight_packed:   [num_experts, K/8, N]
        down_scales:          [num_experts, K/group_size, N]
        shared_gate_up_packed: [K/8, N] (optional shared expert)
        shared_gate_up_scales: [K/group_size, N]
        shared_down_packed:    [K/8, N]
        shared_down_scales:    [K/group_size, N]
    """

    def __init__(
        self,
        gate_up_weight_packed: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_weight_packed: torch.Tensor,
        down_scales: torch.Tensor,
        shared_gate_up_packed: torch.Tensor | None = None,
        shared_gate_up_scales: torch.Tensor | None = None,
        shared_down_packed: torch.Tensor | None = None,
        shared_down_scales: torch.Tensor | None = None,
        *,
        bits: int,
        group_size: int,
        format: str,
        shared_expert_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.format = format
        self.shared_expert_weight = float(shared_expert_weight)
        self.num_experts = int(gate_up_weight_packed.shape[0])

        self.register_buffer("gate_up_weight_packed", gate_up_weight_packed)
        self.register_buffer("gate_up_scales", gate_up_scales)
        self.register_buffer("down_weight_packed", down_weight_packed)
        self.register_buffer("down_scales", down_scales)
        self.register_buffer("shared_gate_up_packed", shared_gate_up_packed)
        self.register_buffer("shared_gate_up_scales", shared_gate_up_scales)
        self.register_buffer("shared_down_packed", shared_down_packed)
        self.register_buffer("shared_down_scales", shared_down_scales)

    @property
    def has_shared_expert(self) -> bool:
        return (
            self.shared_gate_up_packed is not None
            and self.shared_gate_up_scales is not None
            and self.shared_down_packed is not None
            and self.shared_down_scales is not None
        )

    def _infer_shared_group_size(self, hidden_dim: int, intermediate: int) -> int:
        gate_scales = self.shared_gate_up_scales
        down_scales = self.shared_down_scales
        if gate_scales is None or down_scales is None:
            raise RuntimeError("Shared expert scales are missing")
        gate_groups = int(gate_scales.shape[0])
        down_groups = int(down_scales.shape[0])
        if gate_groups <= 0 or hidden_dim % gate_groups != 0:
            raise ValueError("shared_gate_up_scales shape mismatch")
        if down_groups <= 0 or intermediate % down_groups != 0:
            raise ValueError("shared_down_scales shape mismatch")
        gate_group_size = hidden_dim // gate_groups
        down_group_size = intermediate // down_groups
        if gate_group_size != down_group_size:
            raise ValueError("shared expert group sizes disagree")
        return gate_group_size

    def _shared_expert_forward(
        self,
        hidden_states: torch.Tensor,
        *,
        shared_expert_weight: float | None = None,
    ) -> torch.Tensor:
        if not self.has_shared_expert:
            raise RuntimeError("MetalQuantizedMoE has no shared expert weights")

        from .kernels import moe_shared_expert_fp4

        weight = self.shared_expert_weight if shared_expert_weight is None else shared_expert_weight
        hidden_dim = int(hidden_states.shape[-1])
        gate_out = int(self.shared_gate_up_packed.shape[1])
        if gate_out % 2 != 0:
            raise ValueError("shared_gate_up_packed output dim must be even")
        intermediate = gate_out // 2
        group_size = self._infer_shared_group_size(hidden_dim, intermediate)

        return moe_shared_expert_fp4(
            hidden_states,
            self.shared_gate_up_packed,
            self.shared_gate_up_scales,
            self.shared_down_packed,
            self.shared_down_scales,
            group_size=group_size,
            shared_prob=float(weight),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor | None = None,
        expert_probs: torch.Tensor | None = None,
        *,
        shared_expert_weight: float | None = None,
    ) -> torch.Tensor:
        if expert_ids is None or expert_probs is None:
            raise RuntimeError(
                "MetalQuantizedMoE requires expert_ids and expert_probs for routed experts."
            )

        from .moe_ops import fused_moe_forward

        output = fused_moe_forward(
            hidden_states,
            self.gate_up_weight_packed,
            self.gate_up_scales,
            self.down_weight_packed,
            self.down_scales,
            expert_ids,
            expert_probs,
        )

        if self.has_shared_expert:
            shared_out = self._shared_expert_forward(
                hidden_states,
                shared_expert_weight=shared_expert_weight,
            )
            output = output + shared_out

        return output


def find_linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Find all nn.Linear layers in a model.

    Args:
        model: PyTorch model to scan.

    Returns:
        Mapping of layer name -> nn.Linear module.
    """
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def find_moe_layers(model: nn.Module) -> dict[str, MoEModule]:
    """Find MoE layers that store fused expert weights as 3D tensors.

    Args:
        model: PyTorch model to scan.

    Returns:
        Mapping of layer name -> MoE module.
    """
    layers: dict[str, MoEModule] = {}
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        if gate_up is None or down is None:
            continue
        gate_weight = _resolve_moe_weight(gate_up)
        down_weight = _resolve_moe_weight(down)
        if gate_weight is None or down_weight is None:
            continue
        if gate_weight.ndim != 3 or down_weight.ndim != 3:
            continue
        if gate_weight.shape[0] != down_weight.shape[0]:
            continue
        layers[name] = module
    return layers


def get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a dotted module path to its parent module and attribute name.

    Args:
        model: Root model.
        name: Dotted module path (e.g., "model.layers.0.mlp").

    Returns:
        Tuple of (parent_module, attribute_name).
    """
    if "." not in name:
        return model, name

    parent_path, attr = name.rsplit(".", 1)
    parent: nn.Module = model
    for part in parent_path.split("."):
        parent = getattr(parent, part)
    return parent, attr


def quantize_linear_layer(
    linear: nn.Linear,
    bits: int,
    group_size: int,
    format: str,
) -> MetalQuantizedLinear:
    """Quantize an nn.Linear layer into MetalQuantizedLinear.

    Uses MR-GPTQ (Hessian-aware) when calibration data is available on the layer;
    otherwise falls back to RTN (round-to-nearest) quantization.

    Notes:
        MetalQuantizedLinear currently dispatches FP4 kernels for 4-bit weights.
        Using "int4" or "nf4" formats requires compatible kernels at inference.

    Args:
        linear: Source nn.Linear module.
        bits: Quantization bit width (2, 4, or 8).
        group_size: Quantization group size along input dimension.
        format: Quantization format ("fp4", "int4", "nf4") for 4-bit quantization.

    Returns:
        Quantized MetalQuantizedLinear layer.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(linear)}")
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits: {bits}. Use 2, 4, or 8.")
    fmt = format.lower()
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use one of {_SUPPORTED_FORMATS}.")

    if bits != 4:
        if fmt != "fp4":
            raise ValueError("Non-4-bit quantization requires format='fp4'.")
        return MetalQuantizedLinear.from_float(linear, bits=bits, group_size=group_size)

    hessian = _extract_hessian(linear)
    hessian_np = _coerce_hessian(hessian) if hessian is not None else None
    use_hadamard = _extract_use_hadamard(linear)

    quantizer = MRGPTQQuantizer(
        bits=4,
        format=QuantizationFormat(fmt),
        group_size=group_size,
        use_hadamard=use_hadamard,
    )

    weight = linear.weight.detach().float().cpu().numpy()
    packed, scales, _meta = quantizer.quantize_layer(
        weight,
        hessian=hessian_np,
        layer_name=getattr(linear, "_metal_marlin_layer_name", ""),
        use_hadamard=use_hadamard,
    )

    out_features, in_features = linear.weight.shape
    has_bias = linear.bias is not None
    layer = MetalQuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        group_size=group_size,
        bias=has_bias,
    )

    # MR-GPTQ packs as [out, in//8]; MetalQuantizedLinear expects [K//8, N].
    packed_t = torch.from_numpy(packed).to(torch.uint32).T
    scales_t = torch.from_numpy(scales).to(torch.float16).T

    if layer._needs_output_slice:
        pad_cols = layer._padded_out_features - out_features
        packed_t = torch.nn.functional.pad(packed_t, (0, pad_cols, 0, 0))
        scales_t = torch.nn.functional.pad(scales_t, (0, pad_cols, 0, 0))

    layer.weight_packed.copy_(packed_t.to("mps"))
    layer.scales.copy_(scales_t.to("mps"))
    if has_bias:
        layer.bias.copy_(linear.bias.detach().half().to("mps"))

    return layer


def quantize_moe_experts(
    moe_layer: MoEModule,
    bits: int,
    group_size: int,
    format: str,
    *,
    hessians: dict[int, np.ndarray] | None = None,
    use_gptq: bool = True,
    max_workers: int | None = None,
) -> MetalQuantizedMoE:
    """Quantize fused MoE expert weights into MetalQuantizedMoE.

    Uses GPTQ with Hessians when provided for high-quality quantization.
    Falls back to RTN when no Hessians available (lower quality).

    Args:
        moe_layer: Module with .experts.gate_up_proj and .experts.down_proj
            fused expert weights (3D tensors).
        bits: Quantization bit width (4 only for MoE experts).
        group_size: Quantization group size along input dimension.
        format: Quantization format ("fp4", "int4", "nf4").
        hessians: Optional dict mapping expert_idx -> Hessian [in_features, in_features].
                  When provided, uses GPTQ algorithm for each expert.
        use_gptq: If True and hessians provided, use GPTQ. If False, use RTN.
        max_workers: Max parallel workers (auto-computed from available RAM if None).

    Returns:
        MetalQuantizedMoE container with packed weights and scales.
    """
    if bits != 4:
        raise ValueError("MoE expert quantization currently supports only 4-bit weights.")
    fmt = format.lower()
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use one of {_SUPPORTED_FORMATS}.")

    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        raise AttributeError("MoE layer has no 'experts' attribute.")

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    gate_weight = _resolve_moe_weight(gate_up)
    down_weight = _resolve_moe_weight(down)
    if gate_weight is None or down_weight is None:
        raise AttributeError("MoE experts must expose gate_up_proj/down_proj weights.")

    if gate_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("MoE expert weights must be 3D tensors [num_experts, out, in].")
    if gate_weight.shape[0] != down_weight.shape[0]:
        raise ValueError("MoE gate_up/down weights disagree on num_experts.")

    _validate_moe_weight_shape(gate_weight, group_size)
    _validate_moe_weight_shape(down_weight, group_size)

    use_hadamard = _extract_use_hadamard(experts) or _extract_use_hadamard(moe_layer)
    layer_name = getattr(moe_layer, "_metal_marlin_layer_name", "")

    # Use parallel quantization (GPTQ if hessians provided, else RTN)
    # gate_up_proj uses input hessians (input_dim x input_dim)
    gate_packed, gate_scales = _quantize_moe_weight_stack_parallel(
        gate_weight,
        group_size,
        max_workers=max_workers,
        hessians=hessians,
        use_gptq=use_gptq,
    )
    # down_proj uses RTN - Hessians would need intermediate activation collection
    # which is architecture-dependent and not worth the complexity
    down_packed, down_scales = _quantize_moe_weight_stack_parallel(
        down_weight,
        group_size,
        max_workers=max_workers,
        hessians=None,  # RTN for down_proj
        use_gptq=False,
    )

    gate_packed_t = torch.from_numpy(gate_packed).to(torch.uint32).to("mps")
    gate_scales_t = torch.from_numpy(gate_scales).to(torch.float16).to("mps")
    down_packed_t = torch.from_numpy(down_packed).to(torch.uint32).to("mps")
    down_scales_t = torch.from_numpy(down_scales).to(torch.float16).to("mps")

    shared_gate_packed_t: torch.Tensor | None = None
    shared_gate_scales_t: torch.Tensor | None = None
    shared_down_packed_t: torch.Tensor | None = None
    shared_down_scales_t: torch.Tensor | None = None
    shared_expert_weight = getattr(moe_layer, "shared_expert_weight", 1.0)
    if isinstance(shared_expert_weight, torch.Tensor):
        shared_expert_weight = float(shared_expert_weight.detach().cpu().item())
    else:
        shared_expert_weight = float(shared_expert_weight)

    shared_gate_weight, shared_down_weight = _resolve_shared_expert_weights(moe_layer, experts)
    if shared_gate_weight is not None and shared_down_weight is not None:
        _validate_shared_weight_shape(shared_gate_weight, group_size, name="shared_gate_up")
        _validate_shared_weight_shape(shared_down_weight, group_size, name="shared_down")

        # Shared expert uses RTN (single expert, not worth GPTQ overhead)
        shared_gate_packed, shared_gate_scales = _quantize_moe_weight_single_fast(
            shared_gate_weight,
            group_size,
        )
        shared_down_packed, shared_down_scales = _quantize_moe_weight_single_fast(
            shared_down_weight,
            group_size,
        )

        shared_gate_packed_t = torch.from_numpy(shared_gate_packed).to(torch.uint32).to("mps")
        shared_gate_scales_t = torch.from_numpy(shared_gate_scales).to(torch.float16).to("mps")
        shared_down_packed_t = torch.from_numpy(shared_down_packed).to(torch.uint32).to("mps")
        shared_down_scales_t = torch.from_numpy(shared_down_scales).to(torch.float16).to("mps")

    return MetalQuantizedMoE(
        gate_packed_t,
        gate_scales_t,
        down_packed_t,
        down_scales_t,
        shared_gate_up_packed=shared_gate_packed_t,
        shared_gate_up_scales=shared_gate_scales_t,
        shared_down_packed=shared_down_packed_t,
        shared_down_scales=shared_down_scales_t,
        bits=bits,
        group_size=group_size,
        format=fmt,
        shared_expert_weight=shared_expert_weight,
    )


def replace_linear_layers(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    format: str = "fp4",
    skip_patterns: list[str] | None = None,
    layer_config: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Replace nn.Linear layers with MetalQuantizedLinear in-place.

    Args:
        model: Any nn.Module (typically from AutoModelForCausalLM).
        bits: Default bit width (4 for FP4/INT4).
        group_size: Default quantization group size.
        format: Quantization format ("fp4", "int4", "nf4") for 4-bit quantization.
        skip_patterns: Layer name patterns to skip (keep as nn.Linear).
        layer_config: Per-layer config overrides, matched by exact layer name.
            Supports keys: "bits", "group_size", "format", "skip", "hessian",
            and "use_hadamard".

    Returns:
        Dict with replacement statistics:
        - replaced_count: int
        - skipped_count: int
        - replaced_layers: list[str]
        - total_params_quantized: int
    """
    skip_patterns = skip_patterns or []
    layer_config = layer_config or {}

    replacements: list[tuple[str, nn.Linear]] = []
    skipped_count = 0

    for name, module in model.named_modules():
        if isinstance(module, MetalQuantizedLinear):
            skipped_count += 1
            continue
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in skip_patterns):
            skipped_count += 1
            continue
        cfg = layer_config.get(name, {})
        if cfg.get("skip", False):
            skipped_count += 1
            continue
        replacements.append((name, module))

    replaced_layers: list[str] = []
    total_params_quantized = 0

    for name, module in replacements:
        cfg = layer_config.get(name, {})
        layer_bits = int(cfg.get("bits", bits))
        layer_group_size = int(cfg.get("group_size", group_size))
        layer_format = str(cfg.get("format", format))

        if "hessian" in cfg:
            setattr(module, "_metal_marlin_hessian", cfg["hessian"])
        if "use_hadamard" in cfg:
            setattr(module, "_metal_marlin_use_hadamard", cfg["use_hadamard"])
        setattr(module, "_metal_marlin_layer_name", name)

        try:
            quantized = quantize_linear_layer(
                module,
                bits=layer_bits,
                group_size=layer_group_size,
                format=layer_format,
            )
        except Exception:
            skipped_count += 1
            _cleanup_layer_overrides(module)
            continue

        parent, attr = get_parent_module(model, name)
        setattr(parent, attr, quantized)

        replaced_layers.append(name)
        total_params_quantized += module.weight.numel()
        if module.bias is not None:
            total_params_quantized += module.bias.numel()

        _cleanup_layer_overrides(module)

    return {
        "replaced_count": len(replaced_layers),
        "skipped_count": skipped_count,
        "replaced_layers": replaced_layers,
        "total_params_quantized": total_params_quantized,
    }


def replace_moe_layers(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    format: str = "fp4",
    skip_patterns: list[str] | None = None,
    calibration_inputs: list[torch.Tensor] | None = None,
    use_gptq: bool = True,
    max_workers: int | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Replace MoE expert blocks with MetalQuantizedMoE in-place.

    When calibration_inputs provided, collects per-expert Hessians and uses
    GPTQ algorithm for high-quality quantization. Without calibration, falls
    back to RTN (much lower quality for 4-bit).

    Args:
        model: Model to quantize
        bits: Quantization bits (4 only)
        group_size: Quantization group size
        format: Quantization format ("fp4", "int4", "nf4")
        skip_patterns: Module name patterns to skip
        calibration_inputs: List of input_ids tensors for Hessian collection
        use_gptq: Use GPTQ when hessians available (default True)
        max_workers: Max parallel workers (auto from RAM if None)
        device: Device for calibration forward passes
        verbose: Print progress
    """
    skip_patterns = skip_patterns or []

    # Collect expert-level Hessians if calibration data provided
    expert_hessians: dict[str, dict[int, np.ndarray]] = {}
    if calibration_inputs is not None and use_gptq:
        if verbose:
            print(f"Collecting Hessians from {len(calibration_inputs)} calibration samples...")
        expert_hessians = collect_moe_expert_hessians(model, calibration_inputs, device=device)
        if verbose:
            print(f"  Collected Hessians for {len(expert_hessians)} MoE layers")

    replacements: list[tuple[str, MoEModule]] = []
    skipped_count = 0

    for name, module in find_moe_layers(model).items():
        if any(pattern in name for pattern in skip_patterns):
            skipped_count += 1
            continue
        replacements.append((name, module))

    replaced_layers: list[str] = []
    total_params_quantized = 0

    for name, module in replacements:
        experts = getattr(module, "experts", None)
        gate_weight = _resolve_moe_weight(getattr(experts, "gate_up_proj", None))
        down_weight = _resolve_moe_weight(getattr(experts, "down_proj", None))
        if gate_weight is None or down_weight is None:
            skipped_count += 1
            continue

        setattr(module, "_metal_marlin_layer_name", name)

        # Get Hessians for this layer if available
        layer_hessians = expert_hessians.get(name)

        try:
            quantized = quantize_moe_experts(
                module,
                bits=bits,
                group_size=group_size,
                format=format,
                hessians=layer_hessians,
                use_gptq=use_gptq and (layer_hessians is not None),
                max_workers=max_workers,
            )
        except Exception as e:
            if verbose:
                print(f"  Failed to quantize {name}: {e}")
            skipped_count += 1
            _cleanup_layer_overrides(module)
            continue

        # Preserve original experts for potential restoration.
        setattr(module, "_metal_marlin_experts_fp16", experts)
        setattr(module, "experts", quantized)

        replaced_layers.append(name)
        total_params_quantized += gate_weight.numel() + down_weight.numel()

        if verbose:
            method = "GPTQ" if layer_hessians is not None else "RTN"
            print(f"  {name}: {gate_weight.numel() / 1e6:.1f}M params ({method})")

        _cleanup_layer_overrides(module)

    return {
        "replaced_count": len(replaced_layers),
        "skipped_count": skipped_count,
        "replaced_layers": replaced_layers,
        "total_params_quantized": total_params_quantized,
    }


def _extract_hessian(module: nn.Module) -> Any | None:
    for attr in ("_metal_marlin_hessian", "hessian", "calibration_hessian", "hessian_info"):
        if hasattr(module, attr):
            value = getattr(module, attr)
            if value is not None:
                return value
    return None


def _coerce_hessian(value: Any) -> np.ndarray:
    if hasattr(value, "hessian"):
        value = value.hessian
    if isinstance(value, dict) and "hessian" in value:
        value = value["hessian"]
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    raise TypeError(f"Unsupported hessian type: {type(value)}")


def _extract_use_hadamard(module: nn.Module) -> bool:
    value = getattr(module, "_metal_marlin_use_hadamard", None)
    if value is None:
        return False
    return bool(value)


def _cleanup_layer_overrides(module: nn.Module) -> None:
    for attr in ("_metal_marlin_hessian", "_metal_marlin_use_hadamard", "_metal_marlin_layer_name"):
        if hasattr(module, attr):
            delattr(module, attr)


def _resolve_moe_weight(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "weight") and isinstance(value.weight, torch.Tensor):
        return value.weight
    return None


def _validate_moe_weight_shape(weight: torch.Tensor, group_size: int) -> None:
    if weight.ndim != 3:
        raise ValueError("MoE expert weights must be 3D tensors [num_experts, out, in].")
    _num_experts, _out_features, in_features = weight.shape
    if in_features % 8 != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by 8 for MoE packing")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def _validate_shared_weight_shape(
    weight: torch.Tensor,
    group_size: int,
    *,
    name: str,
) -> None:
    if weight.ndim != 2:
        raise ValueError(f"{name} weight must be 2D [out, in]")
    _out_features, in_features = weight.shape
    if in_features % 8 != 0:
        raise ValueError(f"{name} in_features ({in_features}) must be divisible by 8")
    if in_features % group_size != 0:
        raise ValueError(
            f"{name} in_features ({in_features}) must be divisible by group_size ({group_size})"
        )


def _quantize_moe_weight_stack(
    weight: torch.Tensor,
    quantizer: MRGPTQQuantizer,
    *,
    layer_name: str,
    use_hadamard: bool,
) -> tuple[np.ndarray, np.ndarray]:
    weight_np = weight.detach().float().cpu().numpy()
    num_experts, out_features, in_features = weight_np.shape

    packed_all = np.zeros((num_experts, in_features // 8, out_features), dtype=np.uint32)
    scales_all = np.zeros(
        (num_experts, in_features // quantizer.group_size, out_features), dtype=np.float16
    )

    for idx in range(num_experts):
        packed, scales, _meta = quantizer.quantize_layer(
            weight_np[idx],
            hessian=None,
            layer_name=f"{layer_name}.expert_{idx}",
            use_hadamard=use_hadamard,
        )
        # MR-GPTQ packs as [out, in//8]; kernels expect [K//8, N]
        packed_all[idx] = packed.T
        scales_all[idx] = scales.T

    return packed_all, scales_all


def _quantize_moe_weight_stack_parallel(
    weight: torch.Tensor,
    group_size: int,
    *,
    max_workers: int | None = None,
    hessians: dict[int, np.ndarray] | None = None,
    use_gptq: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel GPTQ/RTN quantization for MoE expert weights.

    Uses ProcessPoolExecutor for CPU-bound GPTQ with RAM-aware worker count.
    Each expert is quantized independently in parallel.

    Args:
        weight: Expert weights [num_experts, out_features, in_features]
        group_size: Quantization group size along input dimension
        max_workers: Max parallel workers. If None, computed from available RAM.
        hessians: Optional dict mapping expert_idx -> Hessian [in_features, in_features].
                  If provided, uses GPTQ. If None, falls back to RTN.
        use_gptq: If True and hessians provided, use GPTQ algorithm.

    Returns:
        (packed_all, scales_all) where:
            packed_all: [num_experts, K/8, N] uint32
            scales_all: [num_experts, K/group_size, N] float16
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    weight_np = weight.detach().float().cpu().numpy()
    num_experts, out_features, in_features = weight_np.shape

    # Compute RAM-aware worker count
    if max_workers is None:
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_gb = 16.0  # Conservative fallback

        # Memory per expert: weight + workspace + Hessian (if used)
        bytes_per_expert = out_features * in_features * 4  # FP32 weight
        bytes_per_expert += bytes_per_expert * 2  # Workspace
        if hessians is not None:
            bytes_per_expert += in_features * in_features * 8  # FP64 Hessian

        gb_per_expert = bytes_per_expert / (1024**3)
        # Use 70% of available RAM - for GPTQ, numpy releases GIL so we can exceed CPU count
        max_workers = max(1, min(num_experts, int(available_gb * 0.7 / gb_per_expert)))

    packed_all = np.zeros((num_experts, in_features // 8, out_features), dtype=np.uint32)
    scales_all = np.zeros((num_experts, in_features // group_size, out_features), dtype=np.float16)

    if use_gptq and hessians is not None:
        # GPTQ with Hessians - CPU-bound, use ProcessPoolExecutor
        from .mr_gptq import MRGPTQQuantizer, QuantizationFormat

        def quantize_expert_gptq(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
            idx, expert_weight, expert_hessian, gs = args
            quantizer = MRGPTQQuantizer(
                bits=4,
                format=QuantizationFormat.FP4,
                group_size=gs,
                use_hadamard=True,
            )
            packed, scales, _meta = quantizer.quantize_layer(
                expert_weight,
                hessian=expert_hessian,
                layer_name=f"expert_{idx}",
                use_hadamard=True,
            )
            return idx, packed.T, scales.T

        # ThreadPoolExecutor for GPTQ (ProcessPool has pickle overhead)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            args_list = [
                (idx, weight_np[idx], hessians.get(idx), group_size) for idx in range(num_experts)
            ]
            futures = [executor.submit(quantize_expert_gptq, args) for args in args_list]
            for future in as_completed(futures):
                idx, packed, scales = future.result()
                packed_all[idx] = packed
                scales_all[idx] = scales
    else:
        # RTN fallback - use simple quantize_fp4
        from .quantize_fp4 import quantize_fp4

        def quantize_expert_rtn(idx: int) -> tuple[int, np.ndarray, np.ndarray]:
            expert_weight = weight_np[idx]
            packed, scales = quantize_fp4(expert_weight, group_size=group_size, marlin_layout=True)
            return idx, packed, scales

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(quantize_expert_rtn, idx) for idx in range(num_experts)]
            for future in as_completed(futures):
                idx, packed, scales = future.result()
                packed_all[idx] = packed
                scales_all[idx] = scales

    return packed_all, scales_all


def collect_moe_expert_hessians(
    model: nn.Module,
    calibration_inputs: list[torch.Tensor],
    device: str = "cpu",
    damp_ratio: float = 0.01,
) -> dict[str, dict[int, np.ndarray]]:
    """Collect Hessians for MoE expert layers during calibration forward passes.

    Registers hooks on MoE expert layers to accumulate H = X^T @ X per expert
    based on which tokens route to each expert.

    Args:
        model: Model with MoE layers
        calibration_inputs: List of input_ids tensors for calibration
        device: Device for forward pass
        damp_ratio: Damping ratio for Hessian stability

    Returns:
        Dict mapping layer_name -> {expert_idx: Hessian}
    """
    moe_layers = find_moe_layers(model)
    if not moe_layers:
        return {}

    # Storage: layer_name -> expert_idx -> (H_sum, n_samples)
    hessian_accumulators: dict[str, dict[int, tuple[np.ndarray, int]]] = {}
    hooks: list = []

    def make_expert_hook(layer_name: str, num_experts: int, in_features: int):
        """Create hook to accumulate Hessians based on expert routing."""
        # Initialize per-expert Hessians
        if layer_name not in hessian_accumulators:
            hessian_accumulators[layer_name] = {
                idx: (np.zeros((in_features, in_features), dtype=np.float64), 0)
                for idx in range(num_experts)
            }

        def hook(module, inputs, output):
            # Get router output to determine which tokens go to which experts
            router = getattr(module, "gate", None) or getattr(module, "router", None)
            if router is None:
                return

            # Input to MoE is typically (batch, seq, hidden)
            if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                x = inputs[0].detach().float()
                if x.ndim == 3:
                    batch, seq, hidden = x.shape
                    x_flat = x.reshape(-1, hidden)  # [batch*seq, hidden]

                    # Get routing decisions
                    with torch.no_grad():
                        router_logits = router(x_flat)
                        if hasattr(router_logits, "topk_indices"):
                            topk_indices = router_logits.topk_indices
                        elif isinstance(router_logits, tuple):
                            topk_indices = router_logits[0]
                        else:
                            _, topk_indices = torch.topk(router_logits, k=2, dim=-1)

                    # Accumulate Hessian per expert
                    x_np = x_flat.cpu().numpy().astype(np.float64)
                    indices_np = topk_indices.cpu().numpy()

                    for token_idx in range(x_np.shape[0]):
                        for expert_idx in indices_np[token_idx]:
                            expert_idx = int(expert_idx)
                            h_sum, count = hessian_accumulators[layer_name][expert_idx]
                            token_vec = x_np[token_idx : token_idx + 1]  # [1, hidden]
                            h_sum += token_vec.T @ token_vec
                            hessian_accumulators[layer_name][expert_idx] = (h_sum, count + 1)

        return hook

    # Register hooks on MoE layers
    for name, moe in moe_layers.items():
        experts = getattr(moe, "experts", None)
        if experts is None:
            continue

        gate_up = getattr(experts, "gate_up_proj", None)
        if gate_up is None:
            continue

        weight = _resolve_moe_weight(gate_up)
        if weight is None or weight.ndim != 3:
            continue

        num_experts, out_features, in_features = weight.shape
        hook = moe.register_forward_hook(make_expert_hook(name, num_experts, in_features))
        hooks.append(hook)

    # Run calibration forward passes
    model.eval()
    with torch.no_grad():
        for input_ids in calibration_inputs:
            input_ids = input_ids.to(device)
            _ = model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert accumulators to final Hessians with damping
    result: dict[str, dict[int, np.ndarray]] = {}
    for layer_name, expert_hessians in hessian_accumulators.items():
        result[layer_name] = {}
        for expert_idx, (h_sum, count) in expert_hessians.items():
            if count > 0:
                H = h_sum / count
                # Add damping: H + λ * I where λ = damp_ratio * mean(diag(H))
                damp = damp_ratio * np.mean(np.diag(H))
                H += damp * np.eye(H.shape[0])
                result[layer_name][expert_idx] = H.astype(np.float32)
            else:
                # No samples routed to this expert - use identity
                in_features = h_sum.shape[0]
                result[layer_name][expert_idx] = np.eye(in_features, dtype=np.float32)

    return result


def _quantize_moe_weight_single(
    weight: torch.Tensor,
    quantizer: MRGPTQQuantizer,
    *,
    layer_name: str,
    use_hadamard: bool,
) -> tuple[np.ndarray, np.ndarray]:
    weight_np = weight.detach().float().cpu().numpy()
    packed, scales, _meta = quantizer.quantize_layer(
        weight_np,
        hessian=None,
        layer_name=layer_name,
        use_hadamard=use_hadamard,
    )

    return packed.T, scales.T


def _quantize_moe_weight_single_fast(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast RTN-based FP4 quantization for a single weight matrix.

    Args:
        weight: 2D weight [out_features, in_features]
        group_size: Quantization group size

    Returns:
        (packed, scales) in Marlin layout [K/8, N] and [K/group_size, N]
    """
    from .quantize_fp4 import quantize_fp4

    weight_np = weight.detach().float().cpu().numpy()
    packed, scales = quantize_fp4(weight_np, group_size=group_size, marlin_layout=True)
    return packed, scales


def _resolve_shared_expert_weights(
    moe_layer: MoEModule,
    experts: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    candidates = []
    for container in (moe_layer, experts):
        if container is None:
            continue
        for attr in ("shared_expert", "shared_experts"):
            value = getattr(container, attr, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                if value:
                    candidates.append(value[0])
            elif (
                hasattr(value, "__len__")
                and hasattr(value, "__getitem__")
                and not isinstance(value, torch.Tensor)
            ):
                try:
                    if len(value) > 0:
                        candidates.append(value[0])
                except Exception:
                    candidates.append(value)
            else:
                candidates.append(value)

    for candidate in candidates:
        gate_up = _resolve_moe_weight(getattr(candidate, "gate_up_proj", None))
        down = _resolve_moe_weight(getattr(candidate, "down_proj", None))
        if gate_up is not None and down is not None:
            return gate_up, down

        gate = _resolve_moe_weight(getattr(candidate, "gate_proj", None))
        up = _resolve_moe_weight(getattr(candidate, "up_proj", None))
        down = _resolve_moe_weight(getattr(candidate, "down_proj", None))
        if gate is not None and up is not None and down is not None:
            if gate.shape != up.shape:
                raise ValueError("shared gate_proj/up_proj shapes must match")
            gate_up = torch.cat([gate, up], dim=0)
            return gate_up, down

    return None, None
