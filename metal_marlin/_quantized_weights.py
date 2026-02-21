import re
from itertools import permutations
from typing import Any

import torch
from metal_marlin.mmfp4_loader import MMFP4ModelLoader

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _flatten_tensor_tree(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
) -> dict[str, torch.Tensor]:
    if depth > max_depth:
        return {}

    if torch.is_tensor(value):
        return {prefix or "value": value}

    out: dict[str, torch.Tensor] = {}

    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(
                _flatten_tensor_tree(
                    v,
                    prefix=next_prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
        return out

    if isinstance(value, (list, tuple)):
        for idx, v in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(
                _flatten_tensor_tree(
                    v,
                    prefix=next_prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
        return out

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            if k.startswith("_"):
                continue
            next_prefix = f"{prefix}.{k}" if prefix else k
            out.update(
                _flatten_tensor_tree(
                    v,
                    prefix=next_prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )

    return out


def _find_tensor_by_alias(
    tensor_lookup: dict[str, torch.Tensor],
    aliases: tuple[str, ...],
) -> torch.Tensor | None:
    normalized_aliases = tuple(_normalize_key(a) for a in aliases)

    for alias in normalized_aliases:
        tensor = tensor_lookup.get(alias)
        if tensor is not None:
            return tensor

    for key, tensor in tensor_lookup.items():
        for alias in normalized_aliases:
            if key.endswith(alias) or key.endswith(f"_{alias}"):
                return tensor
    return None


def _has_alias(
    tensor_lookup: dict[str, torch.Tensor],
    aliases: tuple[str, ...],
) -> bool:
    return _find_tensor_by_alias(tensor_lookup, aliases) is not None


def _align_for_copy(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    if src.shape == dst.shape:
        return src.contiguous()

    if src.ndim == dst.ndim and src.ndim >= 2:
        transposed = src.transpose(-1, -2)
        if transposed.shape == dst.shape:
            return transposed.contiguous()

    if src.ndim == dst.ndim and 2 <= src.ndim <= 4:
        identity = tuple(range(src.ndim))
        for perm in permutations(range(src.ndim)):
            if perm == identity:
                continue
            if tuple(src.shape[i] for i in perm) == tuple(dst.shape):
                return src.permute(*perm).contiguous()

    raise ValueError(
        f"Cannot align tensor shape {tuple(src.shape)} to destination shape {tuple(dst.shape)}"
    )


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor, device: str) -> None:
    aligned = _align_for_copy(src, dst)
    target_device = dst.device if dst.device.type != "meta" else torch.device(
        device)
    dst.copy_(aligned.to(device=target_device, dtype=dst.dtype))


def _concat_for_target(
    left: torch.Tensor,
    right: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    if left.ndim != right.ndim:
        return None

    for dim in range(left.ndim):
        if left.shape[:dim] + left.shape[dim + 1:] != right.shape[:dim] + right.shape[dim + 1:]:
            continue
        fused = torch.cat((left, right), dim=dim)
        try:
            _align_for_copy(fused, target)
            return fused
        except ValueError:
            continue
    return None


def _apply_quantized_weights(
    model: Any,
    loader: MMFP4ModelLoader,
    device: str,
) -> int:
    """Apply quantized weights from loader to model.

    Safetensors stores weights in [out_features, in_features//pack_factor] format,
    but MetalQuantizedLinear expects [in_features//pack_factor, out_features]
    (K-major packing for Metal kernels). This function handles the transpose.

    Args:
        model: The model with quantized linear layers
        loader: MMFP4ModelLoader with quantized weights
        device: Target device

    Returns:
        Number of weights loaded
    """
    from metal_marlin.inference_metal import MetalQuantizedLinear

    loaded = 0
    for name, module in model.named_modules():
        if not isinstance(module, MetalQuantizedLinear):
            continue

        # Try to find matching weight in loader
        weight_name = f"{name}.weight"
        try:
            qweight, scales = loader.get_quantized_weight(weight_name)
            if qweight is not None and scales is not None:
                # Safetensors: [out_features, in_features//pack_factor]
                # MetalQuantizedLinear: [in_features//pack_factor, out_features]
                # Need to TRANSPOSE both!
                qweight = qweight.T.contiguous()
                scales = scales.T.contiguous()

                # Move to device with correct dtypes
                qweight = qweight.to(device=device, dtype=torch.uint32)
                scales = scales.to(device=device, dtype=torch.float16)

                # Handle padding if needed
                if hasattr(module, "_needs_output_slice") and module._needs_output_slice:
                    pad_cols = module._padded_out_features - module.out_features
                    qweight = torch.nn.functional.pad(
                        qweight, (0, pad_cols, 0, 0))
                    scales = torch.nn.functional.pad(
                        scales, (0, pad_cols, 0, 0))

                # Copy to module
                if hasattr(module, "weight_packed"):
                    module.weight_packed.copy_(qweight)
                if hasattr(module, "scales"):
                    module.scales.copy_(scales)
                loaded += 1
        except Exception:
            # Weight not found in quantized checkpoint, skip
            pass

    return loaded


def _apply_moe_expert_weights(
    model: Any,
    loader: MMFP4ModelLoader,
    device: str,
) -> int:
    """Apply quantized MoE expert weights from loader to model.

    Finds all QuantizedGlm4MoEExperts modules and loads their weights
    from the safetensors checkpoint.

    Returns:
        Number of expert layers loaded
    """
    try:
        from metal_marlin.glm4_expert_loader import load_expert_weights
        from metal_marlin.glm4_moe_experts import QuantizedGlm4MoEExperts
    except ImportError:
        # Keep import-time behavior stable when GLM4 MoE modules are unavailable.
        return 0

    src_aliases = {
        "gate_packed": (
            "gate_packed",
            "gate_qweight",
            "gate_weight_packed",
            "gate_proj_packed",
            "gate_proj_qweight",
            "gate_proj_weight_packed",
            "gate_weights",
        ),
        "gate_scales": (
            "gate_scales",
            "gate_proj_scales",
        ),
        "up_packed": (
            "up_packed",
            "up_qweight",
            "up_weight_packed",
            "up_proj_packed",
            "up_proj_qweight",
            "up_proj_weight_packed",
            "up_weights",
        ),
        "up_scales": (
            "up_scales",
            "up_proj_scales",
        ),
        "down_packed": (
            "down_packed",
            "down_qweight",
            "down_weight_packed",
            "down_proj_packed",
            "down_proj_qweight",
            "down_proj_weight_packed",
            "down_weights",
        ),
        "down_scales": (
            "down_scales",
            "down_proj_scales",
        ),
        "gate_up_packed": (
            "gate_up_packed",
            "gate_up_qweight",
            "gate_up_weight_packed",
            "gate_up_weights",
            "routed_gate_up_packed",
            "routed_gate_up_weight_packed",
        ),
        "gate_up_scales": (
            "gate_up_scales",
            "routed_gate_up_scales",
        ),
    }

    dst_aliases = {
        "gate_packed": src_aliases["gate_packed"],
        "gate_scales": src_aliases["gate_scales"],
        "up_packed": src_aliases["up_packed"],
        "up_scales": src_aliases["up_scales"],
        "down_packed": src_aliases["down_packed"],
        "down_scales": src_aliases["down_scales"],
        "gate_up_packed": src_aliases["gate_up_packed"],
        "gate_up_scales": src_aliases["gate_up_scales"],
    }

    loaded_layers = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, QuantizedGlm4MoEExperts):
            continue

        layer_match = _LAYER_INDEX_RE.search(module_name)
        if layer_match is None:
            continue
        layer_idx = int(layer_match.group(1))

        try:
            loaded_weights = load_expert_weights(loader, layer_idx)
        except Exception:
            continue

        source_tensors = _flatten_tensor_tree(loaded_weights)
        source_lookup: dict[str, torch.Tensor] = {}
        for key, tensor in source_tensors.items():
            source_lookup.setdefault(_normalize_key(key), tensor)

        buffer_lookup: dict[str, torch.Tensor] = {}
        for buffer_name, buffer_tensor in module.named_buffers():
            buffer_lookup.setdefault(
                _normalize_key(buffer_name), buffer_tensor)

        if not source_lookup or not buffer_lookup:
            continue

        copied_any = False

        def copy_alias(logical_name: str) -> bool:
            dst = _find_tensor_by_alias(
                buffer_lookup, dst_aliases[logical_name])
            src = _find_tensor_by_alias(
                source_lookup, src_aliases[logical_name])
            if dst is None or src is None:
                return False
            try:
                _copy_tensor(dst, src, device)
            except ValueError:
                return False
            return True

        copy_results = {
            "gate_packed": copy_alias("gate_packed"),
            "gate_scales": copy_alias("gate_scales"),
            "up_packed": copy_alias("up_packed"),
            "up_scales": copy_alias("up_scales"),
            "down_packed": copy_alias("down_packed"),
            "down_scales": copy_alias("down_scales"),
            "gate_up_packed": copy_alias("gate_up_packed"),
            "gate_up_scales": copy_alias("gate_up_scales"),
        }

        copied_any = any(copy_results.values())

        if _has_alias(buffer_lookup, dst_aliases["gate_up_packed"]) and not copy_results["gate_up_packed"]:
            dst_gate_up_packed = _find_tensor_by_alias(
                buffer_lookup, dst_aliases["gate_up_packed"])
            src_gate_packed = _find_tensor_by_alias(
                source_lookup, src_aliases["gate_packed"])
            src_up_packed = _find_tensor_by_alias(
                source_lookup, src_aliases["up_packed"])
            if (
                dst_gate_up_packed is not None
                and src_gate_packed is not None
                and src_up_packed is not None
            ):
                fused = _concat_for_target(
                    src_gate_packed, src_up_packed, dst_gate_up_packed)
                if fused is not None:
                    try:
                        _copy_tensor(dst_gate_up_packed, fused, device)
                        copied_any = True
                    except ValueError:
                        pass

        if _has_alias(buffer_lookup, dst_aliases["gate_up_scales"]) and not copy_results["gate_up_scales"]:
            dst_gate_up_scales = _find_tensor_by_alias(
                buffer_lookup, dst_aliases["gate_up_scales"])
            src_gate_scales = _find_tensor_by_alias(
                source_lookup, src_aliases["gate_scales"])
            src_up_scales = _find_tensor_by_alias(
                source_lookup, src_aliases["up_scales"])
            if (
                dst_gate_up_scales is not None
                and src_gate_scales is not None
                and src_up_scales is not None
            ):
                fused = _concat_for_target(
                    src_gate_scales, src_up_scales, dst_gate_up_scales)
                if fused is not None:
                    try:
                        _copy_tensor(dst_gate_up_scales, fused, device)
                        copied_any = True
                    except ValueError:
                        pass

        if copied_any:
            loaded_layers += 1

    return loaded_layers
