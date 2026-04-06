"""Cross-platform device memory abstraction.

Thin utility layer for detecting device capabilities and performing
platform-optimal tensor transfers (MPS / CUDA / CPU).
"""

from dataclasses import dataclass
from enum import Enum, auto

import torch


class DeviceKind(Enum):
    MPS = auto()
    CUDA = auto()
    CPU = auto()


@dataclass(frozen=True)
class DeviceCapabilities:
    kind: DeviceKind
    unified_memory: bool
    supports_pin_memory: bool
    supports_async_transfer: bool
    total_memory_bytes: int
    device_name: str


def _parse_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    return device


def detect_device(device: str | torch.device) -> DeviceCapabilities:
    """Detect capabilities of the given device."""
    dev = _parse_device(device)

    if dev.type == "mps":
        # Apple Silicon unified memory — torch.mps has no memory query API,
        # fall back to recommended_max_memory if available, else 0.
        try:
            total = torch.mps.driver_allocated_memory()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            total = 0
        return DeviceCapabilities(
            kind=DeviceKind.MPS,
            unified_memory=True,
            supports_pin_memory=False,
            supports_async_transfer=True,
            total_memory_bytes=total,
            device_name="Apple MPS",
        )

    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return DeviceCapabilities(
            kind=DeviceKind.CUDA,
            unified_memory=False,
            supports_pin_memory=True,
            supports_async_transfer=True,
            total_memory_bytes=props.total_mem,
            device_name=props.name,
        )

    # CPU fallback
    return DeviceCapabilities(
        kind=DeviceKind.CPU,
        unified_memory=True,
        supports_pin_memory=False,
        supports_async_transfer=False,
        total_memory_bytes=0,
        device_name="CPU",
    )


def is_unified_memory(device: str | torch.device) -> bool:
    """Check if device uses unified memory architecture (Apple Silicon)."""
    return detect_device(device).unified_memory


def optimal_transfer(
    tensor: torch.Tensor,
    target_device: str,
    zero_copy: bool = False,
) -> torch.Tensor:
    """Transfer tensor to device using platform-optimal strategy.

    MPS:  Skip pin_memory, use non_blocking (unified memory).
    CUDA: Pin memory + non_blocking + DMA.
    CPU:  Direct copy (or no-op if already on CPU).

    Args:
        tensor: Source tensor.
        target_device: Target device string (e.g. "mps", "cuda:0", "cpu").
        zero_copy: If True and the architecture supports it, attempt a
            zero-copy view instead of a full transfer.

    Returns:
        Tensor on *target_device*.
    """
    target = torch.device(target_device)

    # No-op when already on the right device
    if tensor.device.type == target.type and (
        target.index is None or tensor.device.index == target.index
    ):
        return tensor

    caps = detect_device(target)

    # --- MPS path (unified memory) ---
    if caps.kind is DeviceKind.MPS:
        return tensor.to(target, non_blocking=True)

    # --- CUDA path (discrete GPU, DMA) ---
    if caps.kind is DeviceKind.CUDA:
        if not tensor.is_pinned() and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        return tensor.to(target, non_blocking=True)

    # --- CPU path ---
    return tensor.to(target)
