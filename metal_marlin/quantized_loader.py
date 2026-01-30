from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


@dataclass
class QuantizedTensor:
    """A quantized weight tensor with its metadata."""

    data: torch.Tensor  # Packed quantized data (uint8/uint16)
    scales: torch.Tensor  # Per-group or per-channel scales
    format: str  # "fp4", "int4", "int2", etc.
    group_size: int
    original_shape: tuple[int, ...]
    needs_hadamard: bool = False
    hadamard_matrix: torch.Tensor | None = None


@dataclass
class QuantizedModel:
    """Container for a fully quantized model."""

    config: dict[str, Any]
    weights: dict[str, QuantizedTensor | torch.Tensor]
    # BF16 weights for things like router gates
    bf16_weights: dict[str, torch.Tensor]

    @classmethod
    def load(cls, path: Path | str) -> QuantizedModel:
        """Load a quantized model from safetensors."""
        path = Path(path)
        config_path = path / "quantization_config.json"
        weights_path = path / "model.safetensors"

        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {path}")

        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        tensors = load_file(weights_path)

        scales_map: dict[str, torch.Tensor] = {}
        group_size_map: dict[str, torch.Tensor] = {}
        quant_type_map: dict[str, str] = {}
        hadamard_block_map: dict[str, torch.Tensor] = {}

        def base_from_suffix(name: str, suffix: str) -> str:
            return name[: -len(suffix)]

        for name, tensor in tensors.items():
            if name.endswith(".scales"):
                scales_map[base_from_suffix(name, ".scales")] = tensor
            elif name.endswith("_scales"):
                scales_map[base_from_suffix(name, "_scales")] = tensor
            elif name.endswith(".group_size"):
                group_size_map[base_from_suffix(name, ".group_size")] = tensor
            elif name.endswith("_group_size"):
                group_size_map[base_from_suffix(name, "_group_size")] = tensor
            elif name.endswith(".quant_type"):
                qtype = _decode_quant_type(tensor)
                if qtype is not None:
                    quant_type_map[base_from_suffix(name, ".quant_type")] = qtype
            elif name.endswith("_quant_type"):
                qtype = _decode_quant_type(tensor)
                if qtype is not None:
                    quant_type_map[base_from_suffix(name, "_quant_type")] = qtype
            elif name.endswith(".hadamard_block_size"):
                hadamard_block_map[base_from_suffix(name, ".hadamard_block_size")] = tensor
            elif name.endswith("_hadamard_block_size"):
                hadamard_block_map[base_from_suffix(name, "_hadamard_block_size")] = tensor

        weights: dict[str, QuantizedTensor | torch.Tensor] = {}
        bf16_weights: dict[str, torch.Tensor] = {}

        meta_suffixes = (
            ".scales",
            "_scales",
            ".group_size",
            "_group_size",
            ".quant_type",
            "_quant_type",
            ".hadamard_block_size",
            "_hadamard_block_size",
        )

        for name, tensor in tensors.items():
            if name.endswith(meta_suffixes):
                continue

            base_name = name[:-10] if name.endswith("_quantized") else name

            has_meta = (
                base_name in scales_map
                or base_name in group_size_map
                or base_name in quant_type_map
                or base_name in hadamard_block_map
                or name.endswith("_quantized")
            )

            if has_meta:
                if base_name in weights or base_name in bf16_weights:
                    continue
                scales = scales_map.get(base_name)
                group_size = _get_group_size(group_size_map.get(base_name), config)
                fmt = _normalize_format(quant_type_map.get(base_name) or config.get("format"))
                original_shape = _infer_original_shape(tensor, scales, group_size, fmt)
                needs_hadamard = base_name in hadamard_block_map

                if scales is None:
                    raise ValueError(f"Missing scales for quantized weight: {base_name}")

                weights[base_name] = QuantizedTensor(
                    data=tensor,
                    scales=scales,
                    format=fmt,
                    group_size=group_size,
                    original_shape=original_shape,
                    needs_hadamard=needs_hadamard,
                    hadamard_matrix=None,
                )
                continue

            if tensor.dtype == torch.bfloat16:
                bf16_weights[base_name] = tensor
            else:
                weights[base_name] = tensor

        return cls(config=config, weights=weights, bf16_weights=bf16_weights)


def _decode_quant_type(tensor: torch.Tensor) -> str | None:
    if tensor.numel() == 0:
        return None
    if tensor.dtype not in (torch.uint8, torch.int8):
        return None
    raw = bytes(int(v) for v in tensor.flatten().to(torch.uint8).tolist())
    raw = raw.strip(b"\x00")
    if not raw:
        return None
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore").strip()


def _get_group_size(tensor: torch.Tensor | None, config: dict[str, Any]) -> int:
    if tensor is not None and tensor.numel() > 0:
        return int(tensor.view(-1)[0].item())
    if "group_size" in config:
        return int(config["group_size"])
    return 0


def _normalize_format(fmt: str | None) -> str:
    if not fmt:
        return "fp4"
    fmt = fmt.lower()
    if fmt.startswith("marlin_"):
        fmt = fmt[len("marlin_") :]
    if fmt in {"fp4", "fp4_e2m1"}:
        return "fp4"
    if fmt in {"fp8", "fp8_e4m3"}:
        return "fp8"
    if fmt in {"int4", "int4_sym"}:
        return "int4"
    if fmt in {"int2", "int3", "nf3", "nf2"}:
        return fmt
    return fmt


def _infer_original_shape(
    data: torch.Tensor,
    scales: torch.Tensor | None,
    group_size: int,
    fmt: str,
) -> tuple[int, ...]:
    if scales is not None and scales.dim() == 2 and group_size > 0:
        if scales.shape[0] == data.shape[0]:
            return (int(scales.shape[0]), int(scales.shape[1]) * group_size)
        if scales.shape[0] * group_size == data.shape[0]:
            return (int(scales.shape[0]) * group_size, int(scales.shape[1]))

    pack_factor = 16 if fmt == "int2" else 8
    if data.dim() >= 2:
        return (int(data.shape[0]), int(data.shape[1]) * pack_factor)
    return (int(data.numel()) * pack_factor,)
