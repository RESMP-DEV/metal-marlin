"""Distributed tensor parallelism for Metal Marlin.

This module provides infrastructure for distributing model inference across
multiple devices. Currently supports:

- **Device Mesh**: Abstract representation of device topology for parallelism
- **Tensor Parallelism**: Column/row parallel linear layers with collective ops
- **Pipeline Parallelism**: Layer-wise distribution with micro-batching

Usage:
    from metal_marlin.distributed import (
        DeviceMesh,
        TensorParallelLinear,
        PipelineStage,
    )

    # Create a 2-device mesh
    mesh = DeviceMesh.from_devices(["gpu:0", "gpu:1"])

    # Wrap a linear layer for tensor parallelism
    tp_linear = TensorParallelLinear.from_marlin_linear(
        linear=model.mlp.gate,
        mesh=mesh,
        parallel_dim="column",  # Split output features
    )

Note:
    Multi-GPU support on Apple Silicon is currently limited. This module
    provides the abstraction layer for future multi-die M-series chips
    or external GPU configurations. For verification, a CPU+GPU split
    mode is available via `DeviceMesh.cpu_gpu_split()`.
"""

from .device_mesh import (
    Device,
    DeviceMesh,
    DeviceType,
)
from .pipeline_parallel import (
    PipelineConfig,
    PipelineParallel,
    PipelineStage,
)
from .tensor_parallel import (
    ParallelMode,
    ShardSpec,
    TensorParallelLinear,
    all_gather,
    all_reduce,
    broadcast,
    scatter,
    shard_tensor,
)

__all__ = [
    # Device mesh
    "Device",
    "DeviceMesh",
    "DeviceType",
    # Tensor parallelism
    "ParallelMode",
    "ShardSpec",
    "TensorParallelLinear",
    "all_gather",
    "all_reduce",
    "broadcast",
    "scatter",
    "shard_tensor",
    # Pipeline parallelism
    "PipelineConfig",
    "PipelineParallel",
    "PipelineStage",
]
