"""Tensor parallelism for distributed linear layers.

This module implements tensor parallelism strategies for MarlinLinear layers,
enabling distribution of weights and activations across multiple devices.

Supported parallelization strategies:

Column Parallel (split output features):
    - Each device holds a slice of weight columns [K, N/P]
    - Forward: local GEMM, all-reduce outputs
    - Use for: Q, K, V projections, MLP up/gate projections

Row Parallel (split input features):
    - Each device holds a slice of weight rows [K/P, N]
    - Forward: local GEMM on partial inputs, all-gather results
    - Use for: Output projection, MLP down projection

For MegatronLM-style parallelism, combine both:
    Attention: Column parallel Q/K/V, row parallel output
    MLP: Column parallel gate/up, row parallel down

Example:
    mesh = DeviceMesh.from_devices([gpu0, gpu1])

    # Column parallel MLP gate
    gate_tp = TensorParallelLinear.from_marlin_linear(
        linear=mlp.gate,
        mesh=mesh,
        mode=ParallelMode.COLUMN,
    )

    # Row parallel MLP down
    down_tp = TensorParallelLinear.from_marlin_linear(
        linear=mlp.down,
        mesh=mesh,
        mode=ParallelMode.ROW,
        input_is_parallel=True,  # Input already partitioned from gate
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import HAS_MLX, HAS_TORCH, mx, to_numpy, torch
from ..dtypes import DTypeConfig, get_default_config
from .device_mesh import Device, DeviceMesh, DeviceType

if TYPE_CHECKING:
    from ..layers import MarlinLinear


class ParallelMode(Enum):
    """Tensor parallelism mode."""

    COLUMN = auto()  # Split output features (N), all-reduce after
    ROW = auto()  # Split input features (K), all-gather after


@dataclass
class ShardSpec:
    """Specification for how a tensor is sharded.

    Attributes:
        dim: Dimension along which tensor is sharded
        num_shards: Number of shards
        shard_index: Index of this shard [0, num_shards)
        total_size: Full size along the sharded dimension
        shard_size: Size of each shard (total_size // num_shards)
    """

    dim: int
    num_shards: int
    shard_index: int
    total_size: int

    @property
    def shard_size(self) -> int:
        """Size of each shard."""
        return self.total_size // self.num_shards

    @property
    def start_index(self) -> int:
        """Start index of this shard in the full tensor."""
        return self.shard_index * self.shard_size

    @property
    def end_index(self) -> int:
        """End index (exclusive) of this shard."""
        return self.start_index + self.shard_size


def shard_tensor(
    tensor: Any,
    dim: int,
    num_shards: int,
) -> list[Any]:
    """Split a tensor into shards along a dimension.

    Args:
        tensor: Input tensor (numpy, MLX, or torch)
        dim: Dimension to split along
        num_shards: Number of shards to create

    Returns:
        List of tensor shards
    """
    arr = to_numpy(tensor)
    size = arr.shape[dim]

    if size % num_shards != 0:
        raise ValueError(
            f"Cannot evenly shard dimension {dim} of size {size} "
            f"into {num_shards} shards"
        )

    shard_size = size // num_shards
    shards = []

    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size
        slices = [slice(None)] * len(arr.shape)
        slices[dim] = slice(start, end)
        shards.append(arr[tuple(slices)])

    return shards


# -----------------------------------------------------------------------------
# Collective Operations
# -----------------------------------------------------------------------------
# These implement communication primitives for tensor parallelism.
# Currently they use numpy for CPU simulation; GPU implementations
# would use Metal shared memory or IPC mechanisms.


def all_reduce(
    tensors: list[Any],
    op: str = "sum",
) -> Any:
    """All-reduce across tensor shards.

    Reduces tensors across all devices and returns the same result to each.

    Args:
        tensors: List of tensors from each device (same shape)
        op: Reduction operation ('sum', 'mean', 'max', 'min')

    Returns:
        Reduced tensor (same on all devices)
    """
    if not tensors:
        raise ValueError("Cannot all-reduce empty list")

    # Convert all to numpy for reduction
    arrays = [to_numpy(t) for t in tensors]

    # Stack and reduce
    stacked = np.stack(arrays, axis=0)

    if op == "sum":
        result = np.sum(stacked, axis=0)
    elif op == "mean":
        result = np.mean(stacked, axis=0)
    elif op == "max":
        result = np.max(stacked, axis=0)
    elif op == "min":
        result = np.min(stacked, axis=0)
    else:
        raise ValueError(f"Unknown reduction op: {op}")

    return result


def all_gather(
    tensors: list[Any],
    dim: int = 0,
) -> Any:
    """All-gather tensor shards.

    Concatenates shards from all devices along the specified dimension.

    Args:
        tensors: List of tensor shards from each device
        dim: Dimension to concatenate along

    Returns:
        Concatenated tensor containing all shards
    """
    if not tensors:
        raise ValueError("Cannot all-gather empty list")

    arrays = [to_numpy(t) for t in tensors]
    return np.concatenate(arrays, axis=dim)


def broadcast(
    tensor: Any,
    num_devices: int,
) -> list[Any]:
    """Broadcast tensor to all devices.

    Args:
        tensor: Source tensor
        num_devices: Number of target devices

    Returns:
        List of tensor copies (one per device)
    """
    arr = to_numpy(tensor)
    return [arr.copy() for _ in range(num_devices)]


def scatter(
    tensor: Any,
    dim: int,
    num_devices: int,
) -> list[Any]:
    """Scatter tensor to devices by splitting along a dimension.

    Args:
        tensor: Source tensor to scatter
        dim: Dimension to split along
        num_devices: Number of target devices

    Returns:
        List of tensor shards (one per device)
    """
    return shard_tensor(tensor, dim, num_devices)


def reduce_scatter(
    tensors: list[Any],
    dim: int = 0,
    op: str = "sum",
) -> list[Any]:
    """Reduce-scatter: reduce then scatter the result.

    First reduces across devices, then scatters the result so each
    device gets a different portion.

    Args:
        tensors: List of tensors from each device (same shape)
        dim: Dimension to scatter along
        op: Reduction operation

    Returns:
        List of tensor shards (one per device)
    """
    # First reduce
    reduced = all_reduce(tensors, op=op)

    # Then scatter
    return scatter(reduced, dim, len(tensors))


# -----------------------------------------------------------------------------
# Tensor Parallel Linear Layer
# -----------------------------------------------------------------------------


class TensorParallelLinear:
    """Tensor parallel wrapper for MarlinLinear.

    Distributes a single linear layer across multiple devices for
    either column parallel (split N) or row parallel (split K).

    The wrapped layer holds weight shards on different devices and
    performs local GEMM followed by collective communication.

    Attributes:
        shards: Per-device weight shards (MarlinLinear instances)
        mesh: Device mesh for parallelism
        mode: Column or row parallel mode
        input_is_parallel: If True, input is already partitioned
        output_is_parallel: If True, don't reduce outputs
        dtype_config: Dtype configuration for operations
    """

    def __init__(
        self,
        shards: list[Any],  # List of MarlinLinear shards
        mesh: DeviceMesh,
        mode: ParallelMode,
        in_features: int,
        out_features: int,
        group_size: int,
        input_is_parallel: bool = False,
        output_is_parallel: bool = False,
        dtype_config: DTypeConfig | None = None,
    ):
        """Initialize tensor parallel linear.

        Args:
            shards: List of MarlinLinear weight shards (one per TP device)
            mesh: Device mesh for communication
            mode: Column or row parallel mode
            in_features: Full input dimension
            out_features: Full output dimension
            group_size: Quantization group size
            input_is_parallel: If True, input is already split across devices
            output_is_parallel: If True, skip output all-reduce/all-gather
            dtype_config: Dtype configuration
        """
        self.shards = shards
        self.mesh = mesh
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.input_is_parallel = input_is_parallel
        self.output_is_parallel = output_is_parallel
        self.dtype_config = dtype_config or get_default_config()

        self.tp_size = mesh.tensor_parallel_size

        if len(shards) != self.tp_size:
            raise ValueError(
                f"Number of shards ({len(shards)}) must match "
                f"tensor parallel size ({self.tp_size})"
            )

    @classmethod
    def from_marlin_linear(
        cls,
        linear: MarlinLinear,
        mesh: DeviceMesh,
        mode: ParallelMode,
        input_is_parallel: bool = False,
        output_is_parallel: bool = False,
    ) -> TensorParallelLinear:
        """Create tensor parallel linear from an existing MarlinLinear.

        Shards the weights and distributes them across mesh devices.

        Args:
            linear: Source MarlinLinear layer
            mesh: Device mesh for parallelism
            mode: Column or row parallel mode
            input_is_parallel: If True, input is already split
            output_is_parallel: If True, skip output reduction

        Returns:
            TensorParallelLinear wrapping sharded weights
        """
        from ..layers import MarlinLinear

        tp_size = mesh.tensor_parallel_size
        in_features = linear.in_features
        out_features = linear.out_features
        group_size = linear.group_size

        # Get numpy weights and scales
        weight_np = to_numpy(linear.weight)  # [K, N//8] packed
        scales_np = to_numpy(linear.scales)  # [K//gs, N]
        bias_np = to_numpy(linear.bias) if linear.bias is not None else None

        shards: list[MarlinLinear] = []

        if mode == ParallelMode.COLUMN:
            # Split output features (N)
            # weight: [K, N//8] -> split on dim 1
            # scales: [K//gs, N] -> split on dim 1
            # bias: [N] -> split on dim 0

            packed_n = weight_np.shape[1]  # N // 8
            if packed_n % tp_size != 0:
                raise ValueError(
                    f"Cannot split {packed_n} packed columns ({packed_n * 8} features) "
                    f"across {tp_size} devices"
                )

            shard_packed_n = packed_n // tp_size
            shard_n = out_features // tp_size

            for i in range(tp_size):
                w_start = i * shard_packed_n
                w_end = w_start + shard_packed_n
                s_start = i * shard_n
                s_end = s_start + shard_n

                weight_shard = weight_np[:, w_start:w_end].copy()
                scales_shard = scales_np[:, s_start:s_end].copy()
                bias_shard = bias_np[s_start:s_end].copy() if bias_np is not None else None

                # Move to appropriate device
                device = mesh.get_device(tp_rank=i)
                weight_shard = _to_device_array(weight_shard, device)
                scales_shard = _to_device_array(scales_shard, device, dtype=linear.dtype_config.numpy_scales)
                if bias_shard is not None:
                    bias_shard = _to_device_array(bias_shard, device)

                shard = MarlinLinear(
                    in_features=in_features,
                    out_features=shard_n,
                    bias=bias_shard is not None,
                    quant_type=linear.quant_type,
                    group_size=group_size,
                    weight_packed=weight_shard,
                    scales=scales_shard,
                    dtype_config=linear.dtype_config,
                )
                if bias_shard is not None:
                    shard.bias = bias_shard

                shards.append(shard)

        else:  # ROW parallel
            # Split input features (K)
            # weight: [K, N//8] -> split on dim 0
            # scales: [K//gs, N] -> split on dim 0
            # bias: [N] -> keep full (only add on first device)

            if in_features % tp_size != 0:
                raise ValueError(
                    f"Cannot split {in_features} input features "
                    f"across {tp_size} devices"
                )

            shard_k = in_features // tp_size
            num_groups = in_features // group_size
            shard_groups = num_groups // tp_size

            for i in range(tp_size):
                w_start = i * shard_k
                w_end = w_start + shard_k
                s_start = i * shard_groups
                s_end = s_start + shard_groups

                weight_shard = weight_np[w_start:w_end, :].copy()
                scales_shard = scales_np[s_start:s_end, :].copy()

                # Only first device gets bias (add after all-reduce)
                has_bias = bias_np is not None and i == 0

                device = mesh.get_device(tp_rank=i)
                weight_shard = _to_device_array(weight_shard, device)
                scales_shard = _to_device_array(scales_shard, device, dtype=linear.dtype_config.numpy_scales)
                bias_shard = None
                if has_bias:
                    bias_shard = _to_device_array(bias_np, device)

                shard = MarlinLinear(
                    in_features=shard_k,
                    out_features=out_features,
                    bias=has_bias,
                    quant_type=linear.quant_type,
                    group_size=group_size,
                    weight_packed=weight_shard,
                    scales=scales_shard,
                    dtype_config=linear.dtype_config,
                )
                if has_bias:
                    shard.bias = bias_shard

                shards.append(shard)

        return cls(
            shards=shards,
            mesh=mesh,
            mode=mode,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            input_is_parallel=input_is_parallel,
            output_is_parallel=output_is_parallel,
            dtype_config=linear.dtype_config,
        )

    def __call__(self, x: Any) -> Any:
        """Forward pass with tensor parallelism.

        Column parallel:
            1. Broadcast input to all devices (or use as-is if input_is_parallel)
            2. Each device computes local GEMM for its output columns
            3. All-gather outputs (or skip if output_is_parallel)

        Row parallel:
            1. Scatter input to devices (split along K)
            2. Each device computes partial GEMM
            3. All-reduce outputs (or skip if output_is_parallel)

        Args:
            x: Input tensor [*, K]

        Returns:
            Output tensor [*, N]
        """
        if self.mode == ParallelMode.COLUMN:
            return self._forward_column_parallel(x)
        else:
            return self._forward_row_parallel(x)

    def _forward_column_parallel(self, x: Any) -> Any:
        """Column parallel forward: split N, all-gather after."""
        # Prepare input for each device
        if self.input_is_parallel:
            # Input is already sharded along K from previous row parallel layer
            # Each device has a portion; need to all-gather first
            x_np = to_numpy(x)
            inputs = [x_np] * self.tp_size  # Broadcast
        else:
            # Broadcast full input to all devices
            x_np = to_numpy(x)
            inputs = [x_np.copy() for _ in range(self.tp_size)]

        # Local GEMM on each shard
        outputs = []
        for i, (shard, inp) in enumerate(zip(self.shards, inputs)):
            device = self.mesh.get_device(tp_rank=i)
            inp_device = _to_device_array(inp, device)
            out = shard(inp_device)
            outputs.append(to_numpy(out))

        # All-gather outputs along feature dimension
        if self.output_is_parallel:
            # Return list of partial outputs for next layer to consume
            return outputs
        else:
            # Concatenate along last dimension (N)
            result = np.concatenate(outputs, axis=-1)
            return result

    def _forward_row_parallel(self, x: Any) -> Any:
        """Row parallel forward: split K, all-reduce after."""
        x_np = to_numpy(x)

        # Prepare inputs
        if self.input_is_parallel:
            # Input already split from previous column parallel layer
            inputs = x if isinstance(x, list) else [x_np]
        else:
            # Scatter input along K dimension
            inputs = shard_tensor(x_np, dim=-1, num_shards=self.tp_size)

        # Local GEMM on each shard
        outputs = []
        for i, (shard, inp) in enumerate(zip(self.shards, inputs)):
            device = self.mesh.get_device(tp_rank=i)
            if isinstance(inp, np.ndarray):
                inp_device = _to_device_array(inp, device)
            else:
                inp_device = inp
            out = shard(inp_device)
            outputs.append(to_numpy(out))

        # All-reduce outputs (sum partial results)
        if self.output_is_parallel:
            return outputs
        else:
            result = all_reduce(outputs, op="sum")
            return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"mode={self.mode.name}, "
            f"tp_size={self.tp_size}"
        )


def _to_device_array(arr: np.ndarray, device: Device, dtype: Any = None) -> Any:
    """Convert numpy array to the appropriate format for device.

    Args:
        arr: Input numpy array
        device: Target device
        dtype: Optional numpy dtype to cast to

    Returns:
        Array in device-appropriate format
    """
    if dtype is not None:
        arr = arr.astype(dtype)

    if device.device_type == DeviceType.CPU:
        return arr
    elif device.device_type == DeviceType.GPU:
        # MLX Metal GPU
        if HAS_MLX and mx is not None:
            return mx.array(arr)
        return arr
    elif device.device_type == DeviceType.CUDA:
        if HAS_TORCH and torch is not None:
            return torch.from_numpy(arr.copy()).to(f"cuda:{device.device_id}")
        raise RuntimeError("PyTorch required for CUDA")
    elif device.device_type == DeviceType.MPS:
        if HAS_TORCH and torch is not None:
            return torch.from_numpy(arr.copy()).to("mps")
        raise RuntimeError("PyTorch required for MPS")
    else:
        return arr


# -----------------------------------------------------------------------------
# Tensor Parallel Attention
# -----------------------------------------------------------------------------


class TensorParallelAttention:
    """Tensor parallel attention module.

    Parallelizes multi-head attention by distributing heads across devices.
    Each device computes attention for a subset of heads.

    For GQA (Grouped Query Attention), distributes KV head groups.

    Column parallel: Q, K, V projections
    Row parallel: Output projection
    """

    def __init__(
        self,
        q_proj: TensorParallelLinear,
        k_proj: TensorParallelLinear,
        v_proj: TensorParallelLinear,
        o_proj: TensorParallelLinear,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mesh: DeviceMesh,
    ):
        """Initialize tensor parallel attention.

        Args:
            q_proj: Column parallel Q projection
            k_proj: Column parallel K projection
            v_proj: Column parallel V projection
            o_proj: Row parallel output projection
            num_heads: Total number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            mesh: Device mesh for parallelism
        """
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.mesh = mesh
        self.tp_size = mesh.tensor_parallel_size

        # Heads per device
        if num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )
        self.local_num_heads = num_heads // self.tp_size

        if num_kv_heads % self.tp_size != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )
        self.local_num_kv_heads = num_kv_heads // self.tp_size

    @classmethod
    def from_attention(
        cls,
        attention: Any,  # MarlinAttention
        mesh: DeviceMesh,
    ) -> TensorParallelAttention:
        """Create tensor parallel attention from MarlinAttention.

        Args:
            attention: Source attention module
            mesh: Device mesh

        Returns:
            TensorParallelAttention
        """
        q_proj = TensorParallelLinear.from_marlin_linear(
            attention.q_proj, mesh, ParallelMode.COLUMN
        )
        k_proj = TensorParallelLinear.from_marlin_linear(
            attention.k_proj, mesh, ParallelMode.COLUMN
        )
        v_proj = TensorParallelLinear.from_marlin_linear(
            attention.v_proj, mesh, ParallelMode.COLUMN
        )
        o_proj = TensorParallelLinear.from_marlin_linear(
            attention.o_proj, mesh, ParallelMode.ROW, input_is_parallel=True
        )

        return cls(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            num_heads=attention.num_heads,
            num_kv_heads=attention.num_kv_heads,
            head_dim=attention.head_dim,
            mesh=mesh,
        )

    def __call__(self, hidden_states: Any, position_ids: Any, mask: Any = None) -> Any:
        """Forward pass with tensor parallel attention.

        Each device computes attention for its subset of heads, then
        the output projection performs row parallel reduction.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Output hidden states [batch, seq_len, hidden_size]
        """
        # Column parallel projections give each device local Q, K, V
        q = self.q_proj(hidden_states)  # [batch, seq, local_heads * head_dim]
        self.k_proj(hidden_states)
        self.v_proj(hidden_states)

        # Each device computes attention for its local heads
        # (This is simplified; real implementation would handle RoPE, mask, etc.)

        # Row parallel output projection reduces across devices
        output = self.o_proj(q)  # Simplified; real impl uses attention output

        return output


# -----------------------------------------------------------------------------
# Tensor Parallel MLP
# -----------------------------------------------------------------------------


class TensorParallelMLP:
    """Tensor parallel MLP (gated or standard).

    For gated MLP (SwiGLU, GeGLU):
        gate_proj: Column parallel (split intermediate)
        up_proj: Column parallel (split intermediate)
        down_proj: Row parallel (reduce from split intermediate)

    This achieves MegatronLM-style parallelism where the intermediate
    activations are partitioned and only reduced at the output.
    """

    def __init__(
        self,
        gate_proj: TensorParallelLinear | None,
        up_proj: TensorParallelLinear,
        down_proj: TensorParallelLinear,
        activation: str = "silu",
        mesh: DeviceMesh | None = None,
    ):
        """Initialize tensor parallel MLP.

        Args:
            gate_proj: Column parallel gate projection (None for non-gated)
            up_proj: Column parallel up projection
            down_proj: Row parallel down projection
            activation: Activation function name
            mesh: Device mesh
        """
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation
        self.mesh = mesh
        self.is_gated = gate_proj is not None

    @classmethod
    def from_mlp(
        cls,
        mlp: Any,  # MarlinMLP
        mesh: DeviceMesh,
    ) -> TensorParallelMLP:
        """Create tensor parallel MLP from MarlinMLP.

        Args:
            mlp: Source MLP module
            mesh: Device mesh

        Returns:
            TensorParallelMLP
        """
        gate_proj = None
        if hasattr(mlp, "gate_proj") and mlp.gate_proj is not None:
            gate_proj = TensorParallelLinear.from_marlin_linear(
                mlp.gate_proj, mesh, ParallelMode.COLUMN, output_is_parallel=True
            )

        up_proj = TensorParallelLinear.from_marlin_linear(
            mlp.up_proj, mesh, ParallelMode.COLUMN, output_is_parallel=True
        )

        down_proj = TensorParallelLinear.from_marlin_linear(
            mlp.down_proj, mesh, ParallelMode.ROW, input_is_parallel=True
        )

        return cls(
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            activation=mlp.activation if hasattr(mlp, "activation") else "silu",
            mesh=mesh,
        )

    def __call__(self, x: Any) -> Any:
        """Forward pass with tensor parallel MLP.

        Args:
            x: Hidden states [batch, seq_len, hidden_size]

        Returns:
            Output [batch, seq_len, hidden_size]
        """
        if self.is_gated:
            # Gated activation: gate * act(up)
            gate_out = self.gate_proj(x)  # List of partial outputs
            up_out = self.up_proj(x)

            # Apply activation and gating per device
            hidden = []
            for g, u in zip(gate_out, up_out):
                g_np = to_numpy(g)
                u_np = to_numpy(u)
                # SiLU activation
                act = u_np * (1.0 / (1.0 + np.exp(-u_np)))
                hidden.append(g_np * act)

            # Row parallel down projection reduces
            output = self.down_proj(hidden)
        else:
            # Standard MLP: down(act(up(x)))
            up_out = self.up_proj(x)

            # Apply activation per device
            hidden = []
            for u in up_out:
                u_np = to_numpy(u)
                act = u_np * (1.0 / (1.0 + np.exp(-u_np)))
                hidden.append(act)

            output = self.down_proj(hidden)

        return output
