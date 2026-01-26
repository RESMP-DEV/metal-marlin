"""Pipeline parallelism for layer-wise model distribution.

This module implements pipeline parallelism where transformer layers are
distributed across devices in sequence. Each device processes its assigned
layers, then passes activations to the next device.

Key concepts:

- **Pipeline Stages**: Groups of layers assigned to a device
- **Micro-batching**: Split batch into smaller chunks for better utilization
- **Bubble**: Idle time at pipeline startup/end

Scheduling strategies:

1. **1F1B (One Forward One Backward)**: Interleave forward/backward passes
   - Forward micro-batch 1, Forward 2, ..., Backward 1, Forward N, Backward 2...
   - Minimizes memory usage by limiting in-flight activations

2. **GPipe**: All forwards, then all backwards
   - Simpler but requires storing all activations

For inference (no backward), pipeline parallelism is straightforward:
each stage processes in order, with activations flowing through.

Example:
    # 4 layers across 2 devices
    mesh = DeviceMesh.from_devices(
        [gpu0, gpu1],
        pipeline_parallel_size=2,
    )

    pipeline = PipelineParallel(
        stages=[
            PipelineStage(layers=[layer0, layer1], device=gpu0),
            PipelineStage(layers=[layer2, layer3], device=gpu1),
        ],
        mesh=mesh,
        num_microbatches=4,
    )

    output = pipeline(input_tokens)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import HAS_MLX, mx, to_numpy
from .device_mesh import Device, DeviceMesh, move_to_device

if TYPE_CHECKING:
    pass


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism.

    Attributes:
        num_microbatches: Number of micro-batches to split the batch into.
            More micro-batches = better utilization but more communication.
        schedule: Pipeline schedule ('1f1b' or 'gpipe')
        overlap_comm: If True, overlap communication with computation
        checkpoint_activations: If True, recompute activations during backward
    """

    num_microbatches: int = 4
    schedule: str = "gpipe"  # 'gpipe' or '1f1b'
    overlap_comm: bool = True
    checkpoint_activations: bool = False

    def __post_init__(self) -> None:
        if self.schedule not in ("gpipe", "1f1b"):
            raise ValueError(f"Unknown schedule: {self.schedule}")
        if self.num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")


@dataclass
class PipelineStage:
    """A single stage in the pipeline.

    Each stage contains one or more transformer layers and is assigned
    to a specific device. During forward pass, the stage processes its
    layers in order and outputs activations to the next stage.

    Attributes:
        layers: List of modules (transformer blocks) in this stage
        device: Device this stage runs on
        stage_id: Index of this stage in the pipeline
        is_first: True if this is the first stage (handles embeddings)
        is_last: True if this is the last stage (handles LM head)
    """

    layers: list[Any]
    device: Device
    stage_id: int = 0
    is_first: bool = False
    is_last: bool = False

    # Optional embedding and LM head for first/last stages
    embedding: Any | None = None
    ln_f: Any | None = None  # Final layer norm
    lm_head: Any | None = None

    def forward(self, hidden_states: Any, **kwargs: Any) -> Any:
        """Process input through all layers in this stage.

        Args:
            hidden_states: Input activations [batch, seq_len, hidden_size]
            **kwargs: Additional arguments (position_ids, mask, etc.)

        Returns:
            Output activations (or logits if is_last)
        """
        # Move input to this device
        x = move_to_device(hidden_states, self.device)

        # Embedding if first stage
        if self.is_first and self.embedding is not None:
            x = self.embedding(x)

        # Process through layers
        for layer in self.layers:
            x = layer(x, **kwargs)

        # LM head if last stage
        if self.is_last:
            if self.ln_f is not None:
                x = self.ln_f(x)
            if self.lm_head is not None:
                x = self.lm_head(x)

        return x

    def __repr__(self) -> str:
        return (
            f"PipelineStage(id={self.stage_id}, "
            f"layers={len(self.layers)}, "
            f"device={self.device})"
        )


class PipelineParallel:
    """Pipeline parallel wrapper for transformer models.

    Distributes model layers across devices in a pipeline topology.
    Supports micro-batching for improved utilization during inference.

    Attributes:
        stages: Ordered list of pipeline stages
        mesh: Device mesh for the pipeline
        config: Pipeline configuration
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        mesh: DeviceMesh,
        config: PipelineConfig | None = None,
    ):
        """Initialize pipeline parallel.

        Args:
            stages: List of PipelineStage in order
            mesh: Device mesh with pipeline_parallel_size >= len(stages)
            config: Pipeline configuration
        """
        self.stages = stages
        self.mesh = mesh
        self.config = config or PipelineConfig()

        if len(stages) != mesh.pipeline_parallel_size:
            raise ValueError(
                f"Number of stages ({len(stages)}) must match "
                f"pipeline parallel size ({mesh.pipeline_parallel_size})"
            )

        # Set stage metadata
        for i, stage in enumerate(stages):
            stage.stage_id = i
            stage.is_first = i == 0
            stage.is_last = i == len(stages) - 1

    @classmethod
    def from_model(
        cls,
        model: Any,
        mesh: DeviceMesh,
        config: PipelineConfig | None = None,
    ) -> PipelineParallel:
        """Create pipeline parallel from a transformer model.

        Automatically partitions layers across pipeline stages based
        on the mesh's pipeline_parallel_size.

        Args:
            model: Transformer model with .layers attribute
            mesh: Device mesh for parallelism
            config: Pipeline configuration

        Returns:
            PipelineParallel wrapping the model
        """
        num_stages = mesh.pipeline_parallel_size
        layers = list(model.layers) if hasattr(model, "layers") else []

        if len(layers) < num_stages:
            raise ValueError(
                f"Cannot create {num_stages} pipeline stages from "
                f"{len(layers)} layers"
            )

        # Evenly distribute layers
        layers_per_stage = len(layers) // num_stages
        remainder = len(layers) % num_stages

        stages: list[PipelineStage] = []
        layer_idx = 0

        for stage_id in range(num_stages):
            # Extra layer for first `remainder` stages
            n_layers = layers_per_stage + (1 if stage_id < remainder else 0)
            stage_layers = layers[layer_idx : layer_idx + n_layers]
            layer_idx += n_layers

            device = mesh.get_pp_group()[stage_id]

            stage = PipelineStage(
                layers=stage_layers,
                device=device,
                stage_id=stage_id,
            )

            # First stage gets embedding
            if stage_id == 0 and hasattr(model, "embedding"):
                stage.embedding = model.embedding

            # Last stage gets LM head
            if stage_id == num_stages - 1:
                if hasattr(model, "ln_f"):
                    stage.ln_f = model.ln_f
                if hasattr(model, "lm_head"):
                    stage.lm_head = model.lm_head

            stages.append(stage)

        return cls(stages=stages, mesh=mesh, config=config)

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """Forward pass through the pipeline.

        Splits input into micro-batches, processes through all stages,
        then concatenates outputs.

        Args:
            inputs: Input tensor (token IDs or embeddings)
            **kwargs: Additional arguments for layers

        Returns:
            Output tensor (logits if model has LM head)
        """
        if self.config.schedule == "gpipe":
            return self._forward_gpipe(inputs, **kwargs)
        else:
            return self._forward_1f1b(inputs, **kwargs)

    def _forward_gpipe(self, inputs: Any, **kwargs: Any) -> Any:
        """GPipe-style forward: all micro-batches through each stage.

        Stage 0: [mb0, mb1, mb2, mb3]
        Stage 1: [mb0, mb1, mb2, mb3]
        ...
        """
        # Split into micro-batches
        inputs_np = to_numpy(inputs)
        batch_size = inputs_np.shape[0]
        num_mb = min(self.config.num_microbatches, batch_size)

        # Evenly divide batch
        mb_size = batch_size // num_mb
        microbatches = []
        for i in range(num_mb):
            start = i * mb_size
            end = start + mb_size if i < num_mb - 1 else batch_size
            microbatches.append(inputs_np[start:end])

        # Process through stages
        for stage in self.stages:
            next_batches = []
            for mb in microbatches:
                out = stage.forward(mb, **kwargs)
                next_batches.append(to_numpy(out))
            microbatches = next_batches

        # Concatenate outputs
        output = np.concatenate(microbatches, axis=0)

        # Convert back to MLX if needed
        if HAS_MLX and mx is not None:
            output = mx.array(output)

        return output

    def _forward_1f1b(self, inputs: Any, **kwargs: Any) -> Any:
        """1F1B-style forward: interleaved processing (inference-only).

        For inference without backward passes, this degenerates to
        streaming micro-batches through the pipeline.

        This implementation is a simplified version; full 1F1B would
        interleave forward and backward passes for training.
        """
        # For inference, 1F1B is similar to GPipe
        # The real benefit of 1F1B is memory savings during training
        return self._forward_gpipe(inputs, **kwargs)

    def _forward_streaming(self, inputs: Any, **kwargs: Any) -> Any:
        """Streaming forward for inference with minimal latency.

        Processes one micro-batch at a time through all stages,
        yielding output as soon as it's available.
        """
        inputs_np = to_numpy(inputs)
        batch_size = inputs_np.shape[0]
        num_mb = min(self.config.num_microbatches, batch_size)
        mb_size = batch_size // num_mb

        outputs = []

        for i in range(num_mb):
            start = i * mb_size
            end = start + mb_size if i < num_mb - 1 else batch_size
            mb = inputs_np[start:end]

            # Process through all stages
            x = mb
            for stage in self.stages:
                x = stage.forward(x, **kwargs)
                x = to_numpy(x)

            outputs.append(x)

        output = np.concatenate(outputs, axis=0)

        if HAS_MLX and mx is not None:
            output = mx.array(output)

        return output

    def get_stage_memory_usage(self) -> list[int]:
        """Estimate memory usage per stage.

        Returns:
            List of estimated bytes per stage
        """
        usage = []
        for stage in self.stages:
            total = 0
            for layer in stage.layers:
                if hasattr(layer, "weight"):
                    w = layer.weight
                    # Packed weights: actual memory is shape * itemsize
                    if hasattr(w, "shape"):
                        total += np.prod(w.shape) * 4  # uint32
                if hasattr(layer, "scales"):
                    s = layer.scales
                    if hasattr(s, "shape"):
                        total += np.prod(s.shape) * 2  # fp16
            usage.append(total)
        return usage

    def __repr__(self) -> str:
        lines = [
            f"PipelineParallel(num_stages={len(self.stages)}, "
            f"schedule={self.config.schedule!r}):"
        ]
        for stage in self.stages:
            lines.append(f"  {stage}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Utility functions for pipeline parallelism
# -----------------------------------------------------------------------------


def assign_layers_to_stages(
    num_layers: int,
    num_stages: int,
    strategy: str = "uniform",
) -> list[tuple[int, int]]:
    """Assign layer ranges to pipeline stages.

    Args:
        num_layers: Total number of layers
        num_stages: Number of pipeline stages
        strategy: Assignment strategy
            - "uniform": Equal layers per stage
            - "memory_balanced": Balance based on estimated memory

    Returns:
        List of (start_layer, end_layer) tuples for each stage
    """
    if strategy == "uniform":
        layers_per_stage = num_layers // num_stages
        remainder = num_layers % num_stages

        ranges = []
        start = 0
        for i in range(num_stages):
            n = layers_per_stage + (1 if i < remainder else 0)
            ranges.append((start, start + n))
            start += n
        return ranges

    elif strategy == "memory_balanced":
        # For memory balancing, we'd need actual weight sizes
        # Fall back to uniform for now
        return assign_layers_to_stages(num_layers, num_stages, "uniform")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def estimate_pipeline_bubble(
    num_stages: int,
    num_microbatches: int,
    time_per_stage: float = 1.0,
) -> float:
    """Estimate pipeline bubble overhead.

    The pipeline bubble occurs at the start (warmup) and end (cooldown)
    when not all stages are active.

    Args:
        num_stages: Number of pipeline stages
        num_microbatches: Number of micro-batches
        time_per_stage: Time to process one micro-batch in one stage

    Returns:
        Bubble time as a fraction of ideal parallel time
    """
    # Total time = warmup + steady state + cooldown
    # Warmup: (P-1) * time_per_stage (filling the pipeline)
    # Steady: M * time_per_stage (processing M micro-batches)
    # Cooldown: (P-1) * time_per_stage (draining)
    #
    # Ideal time = M * time_per_stage (if all stages worked in parallel)
    # Bubble = warmup + cooldown = 2 * (P-1) * time_per_stage

    P = num_stages
    M = num_microbatches

    total_time = M * time_per_stage + 2 * (P - 1) * time_per_stage
    ideal_time = M * time_per_stage

    bubble_fraction = (total_time - ideal_time) / total_time
    return bubble_fraction


def optimal_num_microbatches(
    batch_size: int,
    num_stages: int,
    memory_per_activation: int,
    available_memory: int,
) -> int:
    """Calculate optimal number of micro-batches.

    Balances:
    - Fewer micro-batches = larger bubbles
    - More micro-batches = more communication overhead

    Also considers memory constraints.

    Args:
        batch_size: Total batch size
        num_stages: Number of pipeline stages
        memory_per_activation: Memory for one activation tensor (bytes)
        available_memory: Available memory per device (bytes)

    Returns:
        Optimal number of micro-batches
    """
    # Minimum: at least as many as stages for pipeline utilization
    min_mb = num_stages

    # Maximum based on memory: need to store activations for in-flight batches
    # With 1F1B, in-flight = num_stages
    max_mb_memory = available_memory // (memory_per_activation * num_stages)
    max_mb_memory = max(max_mb_memory, 1)

    # Maximum based on batch size
    max_mb_batch = batch_size

    max_mb = min(max_mb_memory, max_mb_batch)

    # Choose: minimize bubble while respecting constraints
    # Bubble decreases with more micro-batches, so use max
    optimal = max(min_mb, min(max_mb, batch_size))

    return optimal


@dataclass
class PipelineSchedule:
    """A schedule of operations for pipeline execution.

    Each entry specifies: (stage_id, micro_batch_id, operation)
    Operations: 'forward', 'backward', 'send', 'recv'
    """

    steps: list[tuple[int, int, str]] = field(default_factory=list)

    @classmethod
    def gpipe_schedule(
        cls,
        num_stages: int,
        num_microbatches: int,
    ) -> PipelineSchedule:
        """Generate GPipe schedule.

        All forwards first, then all backwards.
        For inference, only forward steps.
        """
        steps = []

        # Forward passes
        for mb in range(num_microbatches):
            for stage in range(num_stages):
                steps.append((stage, mb, "forward"))

        return cls(steps=steps)

    @classmethod
    def one_f_one_b_schedule(
        cls,
        num_stages: int,
        num_microbatches: int,
    ) -> PipelineSchedule:
        """Generate 1F1B schedule.

        Interleaves forward and backward passes to minimize memory.
        For inference-only, degenerates to streaming.
        """
        steps = []

        # Warmup: fill pipeline
        for i in range(num_stages):
            for stage in range(i + 1):
                steps.append((stage, i - stage, "forward"))

        # Steady state: 1F1B
        for mb in range(num_stages, num_microbatches):
            for stage in range(num_stages):
                steps.append((stage, mb - stage, "forward"))

        # Cooldown: drain pipeline
        for i in range(num_stages):
            for stage in range(i, num_stages):
                steps.append((stage, num_microbatches - 1 - (stage - i), "forward"))

        return cls(steps=steps)
