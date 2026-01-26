"""Device mesh abstraction for distributed tensor operations.

A DeviceMesh represents a logical grid of devices that can be used for
tensor parallelism and pipeline parallelism. This module provides:

- Device enumeration and capability detection
- Mesh topology for N-D parallelism (tensor parallel, pipeline parallel)
- Placement strategies for weight and activation sharding

Currently supported device types:
- GPU (Metal via MLX on Apple Silicon)
- CPU (numpy fallback for verification)

Future support planned for:
- Multi-die M-series configurations
- External GPUs via Thunderbolt
- Multi-node distributed inference
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from .._compat import HAS_MLX, HAS_TORCH, mx, torch

if TYPE_CHECKING:
    pass


class DeviceType(Enum):
    """Device type enumeration."""

    CPU = auto()
    GPU = auto()  # Metal GPU (Apple Silicon)
    CUDA = auto()  # NVIDIA GPU (via PyTorch)
    MPS = auto()  # Metal Performance Shaders (PyTorch on macOS)


@dataclass(frozen=True)
class Device:
    """A single compute device.

    Attributes:
        device_type: Type of device (CPU, GPU, CUDA, MPS)
        device_id: Device index (0 for first GPU, etc.)
        name: Human-readable device name
        memory_bytes: Total device memory in bytes (0 if unknown)
    """

    device_type: DeviceType
    device_id: int = 0
    name: str = ""
    memory_bytes: int = 0

    @property
    def is_gpu(self) -> bool:
        """True if this is any type of GPU device."""
        return self.device_type in (DeviceType.GPU, DeviceType.CUDA, DeviceType.MPS)

    @property
    def is_cpu(self) -> bool:
        """True if this is a CPU device."""
        return self.device_type == DeviceType.CPU

    def __str__(self) -> str:
        type_name = self.device_type.name.lower()
        return f"{type_name}:{self.device_id}"

    @classmethod
    def cpu(cls, device_id: int = 0) -> Device:
        """Create a CPU device."""
        import platform

        return cls(
            device_type=DeviceType.CPU,
            device_id=device_id,
            name=platform.processor() or "cpu",
        )

    @classmethod
    def gpu(cls, device_id: int = 0) -> Device:
        """Create a Metal GPU device (Apple Silicon)."""
        name = "Apple Silicon GPU"
        memory = 0

        if HAS_MLX and mx is not None:
            # MLX doesn't expose device enumeration, but we can detect
            # unified memory size via system calls
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                memory = int(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass

        return cls(
            device_type=DeviceType.GPU,
            device_id=device_id,
            name=name,
            memory_bytes=memory,
        )

    @classmethod
    def cuda(cls, device_id: int = 0) -> Device:
        """Create a CUDA device (NVIDIA GPU via PyTorch)."""
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch required for CUDA devices")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        if device_id >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device_id} not found")

        props = torch.cuda.get_device_properties(device_id)
        return cls(
            device_type=DeviceType.CUDA,
            device_id=device_id,
            name=props.name,
            memory_bytes=props.total_memory,
        )

    @classmethod
    def mps(cls, device_id: int = 0) -> Device:
        """Create an MPS device (Metal via PyTorch on macOS)."""
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch required for MPS devices")

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")

        return cls(
            device_type=DeviceType.MPS,
            device_id=device_id,
            name="Apple MPS",
        )


@dataclass
class DeviceMesh:
    """A logical mesh of devices for parallelism.

    The mesh is organized as a 2D grid with dimensions:
    - Axis 0: Tensor parallel dimension (splits within a layer)
    - Axis 1: Pipeline parallel dimension (splits across layers)

    For single-axis parallelism, one dimension will be 1.

    Attributes:
        devices: 2D array of Device objects
        tensor_parallel_size: Size of tensor parallel dimension
        pipeline_parallel_size: Size of pipeline parallel dimension
    """

    devices: list[list[Device]] = field(default_factory=list)

    @property
    def tensor_parallel_size(self) -> int:
        """Number of devices in tensor parallel dimension."""
        return len(self.devices[0]) if self.devices else 0

    @property
    def pipeline_parallel_size(self) -> int:
        """Number of devices in pipeline parallel dimension."""
        return len(self.devices) if self.devices else 0

    @property
    def world_size(self) -> int:
        """Total number of devices in the mesh."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    def __len__(self) -> int:
        return self.world_size

    def __iter__(self) -> Iterator[Device]:
        """Iterate over all devices in row-major order."""
        for row in self.devices:
            yield from row

    def get_device(self, tp_rank: int, pp_rank: int = 0) -> Device:
        """Get device at specified tensor/pipeline parallel ranks.

        Args:
            tp_rank: Tensor parallel rank (column index)
            pp_rank: Pipeline parallel rank (row index)

        Returns:
            Device at the specified position
        """
        return self.devices[pp_rank][tp_rank]

    def get_tp_group(self, pp_rank: int = 0) -> list[Device]:
        """Get all devices in a tensor parallel group.

        Args:
            pp_rank: Pipeline parallel rank (row index)

        Returns:
            List of devices that form a tensor parallel group
        """
        return list(self.devices[pp_rank])

    def get_pp_group(self, tp_rank: int = 0) -> list[Device]:
        """Get all devices in a pipeline parallel group.

        Args:
            tp_rank: Tensor parallel rank (column index)

        Returns:
            List of devices that form a pipeline parallel group
        """
        return [row[tp_rank] for row in self.devices]

    @classmethod
    def from_devices(
        cls,
        devices: list[Device],
        tensor_parallel_size: int | None = None,
        pipeline_parallel_size: int | None = None,
    ) -> DeviceMesh:
        """Create a mesh from a flat list of devices.

        Args:
            devices: List of Device objects
            tensor_parallel_size: TP dimension size. Defaults to len(devices).
            pipeline_parallel_size: PP dimension size. Defaults to 1.

        Returns:
            DeviceMesh with specified topology
        """
        n_devices = len(devices)
        if n_devices == 0:
            return cls(devices=[])

        # Default: all devices in tensor parallel dimension
        if tensor_parallel_size is None and pipeline_parallel_size is None:
            tensor_parallel_size = n_devices
            pipeline_parallel_size = 1
        elif tensor_parallel_size is None:
            assert pipeline_parallel_size is not None
            if n_devices % pipeline_parallel_size != 0:
                raise ValueError(
                    f"Cannot divide {n_devices} devices into {pipeline_parallel_size} PP groups"
                )
            tensor_parallel_size = n_devices // pipeline_parallel_size
        elif pipeline_parallel_size is None:
            if n_devices % tensor_parallel_size != 0:
                raise ValueError(
                    f"Cannot divide {n_devices} devices into {tensor_parallel_size} TP groups"
                )
            pipeline_parallel_size = n_devices // tensor_parallel_size

        if tensor_parallel_size * pipeline_parallel_size != n_devices:
            raise ValueError(
                f"TP size ({tensor_parallel_size}) * PP size ({pipeline_parallel_size}) "
                f"!= device count ({n_devices})"
            )

        # Arrange into 2D grid (PP rows, TP columns)
        mesh: list[list[Device]] = []
        idx = 0
        for _ in range(pipeline_parallel_size):
            row: list[Device] = []
            for _ in range(tensor_parallel_size):
                row.append(devices[idx])
                idx += 1
            mesh.append(row)

        return cls(devices=mesh)

    @classmethod
    def cpu_only(cls, n_devices: int = 1) -> DeviceMesh:
        """Create a mesh with only CPU devices.

        Useful for verification and testing.

        Args:
            n_devices: Number of logical CPU devices to create

        Returns:
            DeviceMesh with n_devices CPU devices
        """
        devices = [Device.cpu(i) for i in range(n_devices)]
        return cls.from_devices(devices)

    @classmethod
    def cpu_gpu_split(cls) -> DeviceMesh:
        """Create a 2-device mesh with CPU and GPU for verification.

        This is useful for testing tensor parallel logic without
        requiring multiple physical GPUs. The CPU device handles
        one shard and the GPU handles another.

        Returns:
            DeviceMesh with [CPU, GPU] tensor parallel topology
        """
        devices = [Device.cpu(0), Device.gpu(0)]
        return cls.from_devices(devices, tensor_parallel_size=2)

    @classmethod
    def single_gpu(cls) -> DeviceMesh:
        """Create a mesh with a single GPU device.

        This is the default for single-GPU inference.

        Returns:
            DeviceMesh with one GPU device
        """
        return cls.from_devices([Device.gpu(0)])

    @classmethod
    def detect_available(cls) -> DeviceMesh:
        """Auto-detect available devices and create a mesh.

        Priority:
        1. MLX Metal GPU (Apple Silicon)
        2. CUDA GPUs (via PyTorch)
        3. MPS (Metal via PyTorch)
        4. CPU fallback

        Returns:
            DeviceMesh with all available devices
        """
        devices: list[Device] = []

        # Try MLX (Apple Silicon GPU)
        if HAS_MLX:
            devices.append(Device.gpu(0))
            return cls.from_devices(devices)

        # Try CUDA
        if HAS_TORCH and torch is not None and torch.cuda.is_available():
            n_cuda = torch.cuda.device_count()
            for i in range(n_cuda):
                devices.append(Device.cuda(i))
            return cls.from_devices(devices)

        # Try MPS
        if HAS_TORCH and torch is not None and torch.backends.mps.is_available():
            devices.append(Device.mps(0))
            return cls.from_devices(devices)

        # Fallback to CPU
        devices.append(Device.cpu(0))
        return cls.from_devices(devices)

    def __repr__(self) -> str:
        lines = [f"DeviceMesh(tp_size={self.tensor_parallel_size}, pp_size={self.pipeline_parallel_size}):"]
        for pp_rank, row in enumerate(self.devices):
            devices_str = ", ".join(str(d) for d in row)
            lines.append(f"  PP rank {pp_rank}: [{devices_str}]")
        return "\n".join(lines)


def get_device_for_array(arr: Any) -> Device:
    """Detect which device an array is on.

    Args:
        arr: Array (numpy, MLX, or torch tensor)

    Returns:
        Device where the array resides
    """
    # MLX arrays
    if HAS_MLX and type(arr).__module__.startswith("mlx"):
        return Device.gpu(0)  # MLX uses unified memory on single GPU

    # PyTorch tensors
    if HAS_TORCH and torch is not None and isinstance(arr, torch.Tensor):
        device = arr.device
        if device.type == "cuda":
            return Device.cuda(device.index or 0)
        elif device.type == "mps":
            return Device.mps(0)
        else:
            return Device.cpu(0)

    # Numpy arrays are always on CPU
    return Device.cpu(0)


def move_to_device(arr: Any, device: Device) -> Any:
    """Move an array to the specified device.

    Args:
        arr: Input array (numpy, MLX, or torch tensor)
        device: Target device

    Returns:
        Array on the target device
    """
    from .._compat import from_numpy, to_numpy

    current_device = get_device_for_array(arr)

    # Already on target device
    if current_device == device:
        return arr

    # Convert to numpy as intermediate
    arr_np = to_numpy(arr)

    # Target is CPU
    if device.device_type == DeviceType.CPU:
        return arr_np

    # Target is MLX GPU
    if device.device_type == DeviceType.GPU:
        return from_numpy(arr_np, backend="mlx")

    # Target is CUDA or MPS (via PyTorch)
    if device.device_type in (DeviceType.CUDA, DeviceType.MPS):
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch required for CUDA/MPS devices")

        tensor = torch.from_numpy(arr_np.copy())

        if device.device_type == DeviceType.CUDA:
            return tensor.to(f"cuda:{device.device_id}")
        else:
            return tensor.to("mps")

    raise ValueError(f"Unknown device type: {device.device_type}")
