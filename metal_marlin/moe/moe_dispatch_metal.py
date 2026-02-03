"""Asynchronous Metal command buffer submission for parallel expert execution.

This module provides Metal-accelerated MoE dispatch with asynchronous command
buffer management for parallel expert execution. It enables:
    - Concurrent execution of multiple experts via separate command buffers
    - Overlapping of CPU dispatch encoding with GPU execution
    - Efficient synchronization for parallel expert workloads

Usage:
    from metal_marlin.moe.moe_dispatch_metal import AsyncExpertCommandBuffer
    from metal_marlin.moe.moe_dispatch_metal import ParallelExpertExecutor

    # Method 1: Direct async command buffer usage
    executor = ParallelExpertExecutor(lib)
    with executor.batch_dispatch() as batch:
        for expert_idx, tokens in expert_assignments:
            batch.dispatch_expert_async(
                expert_idx=expert_idx,
                kernel="moe_expert_gemm",
                grid=(num_threadgroups, 1, 1),
                threadgroup=(128, 1, 1),
                buffers=[input_buf, weight_buf, output_buf],
            )
    # Exits context with automatic commit and wait

    # Method 2: Lower-level command buffer management
    async_buffer = AsyncExpertCommandBuffer(lib)
    async_buffer.begin_encoding()
    async_buffer.encode_kernel(
        kernel_name="moe_expert_gemm",
        grid=(16, 1, 1),
        threadgroup=(128, 1, 1),
        buffers=[buf1, buf2, buf3],
    )
    async_buffer.commit_async()
    async_buffer.wait_for_completion()
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .._compat import Metal
from ..metal_dispatch import (
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Async Command Buffer Management
# -----------------------------------------------------------------------------


@dataclass
class ExpertDispatchJob:
    """Represents a single expert kernel dispatch job.

    Attributes:
        expert_idx: Index of the expert this job is for.
        kernel_name: Name of the Metal kernel function to dispatch.
        grid: Threadgroup grid dimensions (x, y, z).
        threadgroup: Threads per threadgroup (x, y, z).
        buffers: Metal buffers to bind as kernel arguments.
        command_buffer: The Metal command buffer (assigned during encoding).
        encoder: The compute command encoder (assigned during encoding).
    """

    expert_idx: int
    kernel_name: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    buffers: Sequence[Any]
    command_buffer: Any | None = None
    encoder: Any | None = None


class AsyncExpertCommandBuffer:
    """Asynchronous command buffer for single expert execution.

    This class wraps a single Metal command buffer for encoding expert
    kernel dispatches. It supports async submission and completion tracking.

    Usage:
        async_buf = AsyncExpertCommandBuffer(lib)
        async_buf.begin_encoding()
        async_buf.encode_kernel("expert_gemm", (16, 1, 1), (128, 1, 1), buffers)
        async_buf.commit_async()
        # ... do other work ...
        async_buf.wait_for_completion()
    """

    __slots__ = ("_lib", "_command_buffer", "_encoder", "_committed", "_pipeline_cache")

    def __init__(self, lib: Any) -> None:
        """Initialize async command buffer.

        Args:
            lib: MetalKernelLibrary instance for accessing Metal device and pipelines.
        """
        self._lib = lib
        self._command_buffer: Any | None = None
        self._encoder: Any | None = None
        self._committed = False
        self._pipeline_cache: dict[str, Any] = {}

    def _get_pipeline(self, kernel_name: str) -> Any:
        """Get or create compute pipeline state for a kernel."""
        if kernel_name not in self._pipeline_cache:
            self._pipeline_cache[kernel_name] = self._lib.get_pipeline(kernel_name)
        return self._pipeline_cache[kernel_name]

    def begin_encoding(self) -> AsyncExpertCommandBuffer:
        """Begin encoding a new command buffer.

        Returns:
            Self for method chaining.
        """
        if self._command_buffer is not None:
            raise RuntimeError("Command buffer already exists. Call reset() first.")

        self._command_buffer = self._lib.command_queue.commandBuffer()
        self._encoder = self._command_buffer.computeCommandEncoder()
        self._committed = False
        return self

    def encode_kernel(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
    ) -> None:
        """Encode a kernel dispatch into the command buffer.

        Args:
            kernel_name: Name of the Metal kernel function.
            grid: Threadgroup grid dimensions (x, y, z).
            threadgroup: Threads per threadgroup (x, y, z).
            buffers: Metal buffers to bind as kernel arguments.

        Raises:
            RuntimeError: If encoding has not started or already committed.
        """
        if self._encoder is None:
            raise RuntimeError("No active encoder. Call begin_encoding() first.")
        if self._committed:
            raise RuntimeError("Command buffer already committed.")

        pipeline = self._get_pipeline(kernel_name)
        self._encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffers):
            self._encoder.setBuffer_offset_atIndex_(buf, 0, i)

        self._encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )

    def commit_async(self) -> None:
        """Commit the command buffer for asynchronous execution.

        The GPU begins execution immediately after this call returns.
        Call wait_for_completion() to block until execution finishes.
        """
        if self._encoder is None:
            raise RuntimeError("No active encoder.")
        if self._committed:
            raise RuntimeError("Command buffer already committed.")

        self._encoder.endEncoding()
        self._command_buffer.commit()
        self._committed = True

    def wait_for_completion(self) -> None:
        """Block until the command buffer completes execution.

        Raises:
            RuntimeError: If the command buffer was not committed.
        """
        if not self._committed:
            raise RuntimeError("Command buffer not committed. Call commit_async() first.")
        if self._command_buffer is None:
            raise RuntimeError("No command buffer to wait for.")

        self._command_buffer.waitUntilCompleted()

    def is_completed(self) -> bool:
        """Check if command buffer execution has completed.

        Returns:
            True if completed, False otherwise.
        """
        if self._command_buffer is None:
            return False
        return self._command_buffer.status() == Metal.MTLCommandBufferStatusCompleted

    def reset(self) -> None:
        """Reset the command buffer for reuse."""
        self._command_buffer = None
        self._encoder = None
        self._committed = False

    def __enter__(self) -> AsyncExpertCommandBuffer:
        """Context manager entry."""
        self.begin_encoding()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: commit and wait if no exception."""
        if exc_type is None and self._encoder is not None:
            self.commit_async()
            self.wait_for_completion()
        self.reset()


# -----------------------------------------------------------------------------
# Parallel Expert Execution Manager
# -----------------------------------------------------------------------------


class ParallelExpertExecutor:
    """Manager for parallel execution of multiple experts.

    This class enables true parallel expert execution by using separate
    command buffers for each expert. All command buffers are committed
    simultaneously, allowing the GPU to schedule them concurrently.

    Usage:
        executor = ParallelExpertExecutor(lib)

        # Method 1: Using batch context manager
        with executor.batch_dispatch() as batch:
            for i in range(num_experts):
                batch.dispatch_expert_async(
                    expert_idx=i,
                    kernel="moe_expert_gemm",
                    grid=(grids[i], 1, 1),
                    threadgroup=(128, 1, 1),
                    buffers=[input_bufs[i], weight_bufs[i], output_bufs[i]],
                )

        # Method 2: Manual dispatch
        executor.begin_batch()
        for i in range(num_experts):
            executor.begin_expert_command(i)
            executor.encode_expert_kernel(
                kernel_name="moe_expert_gemm",
                grid=(grids[i], 1, 1),
                threadgroup=(128, 1, 1),
                buffers=[input_bufs[i], weight_bufs[i], output_bufs[i]],
            )
        executor.commit_all_async()
        executor.wait_for_all()
    """

    __slots__ = (
        "_lib",
        "_jobs",
        "_active_jobs",
        "_committed",
        "_pipeline_cache",
    )

    def __init__(self, lib: Any) -> None:
        """Initialize parallel expert executor.

        Args:
            lib: MetalKernelLibrary instance.
        """
        self._lib = lib
        self._jobs: dict[int, ExpertDispatchJob] = {}
        self._active_jobs: dict[int, ExpertDispatchJob] = {}
        self._committed = False
        self._pipeline_cache: dict[str, Any] = {}

    def _get_pipeline(self, kernel_name: str) -> Any:
        """Get or create compute pipeline state for a kernel."""
        if kernel_name not in self._pipeline_cache:
            self._pipeline_cache[kernel_name] = self._lib.get_pipeline(kernel_name)
        return self._pipeline_cache[kernel_name]

    def begin_batch(self) -> None:
        """Begin a new batch of expert dispatches."""
        if self._active_jobs:
            raise RuntimeError("Active jobs exist. Call wait_for_all() first.")
        self._jobs.clear()
        self._active_jobs.clear()
        self._committed = False

    def begin_expert_command(self, expert_idx: int) -> None:
        """Begin encoding a command buffer for a specific expert.

        Args:
            expert_idx: Index of the expert.

        Raises:
            RuntimeError: If already committed or expert already has active command.
        """
        if self._committed:
            raise RuntimeError("Batch already committed.")
        if expert_idx in self._active_jobs:
            raise RuntimeError(f"Expert {expert_idx} already has an active command buffer.")

        command_buffer = self._lib.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        job = ExpertDispatchJob(
            expert_idx=expert_idx,
            kernel_name="",
            grid=(0, 0, 0),
            threadgroup=(0, 0, 0),
            buffers=[],
            command_buffer=command_buffer,
            encoder=encoder,
        )
        self._active_jobs[expert_idx] = job

    def encode_expert_kernel(
        self,
        expert_idx: int,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
    ) -> None:
        """Encode a kernel for a specific expert.

        Args:
            expert_idx: Index of the expert.
            kernel_name: Name of the Metal kernel.
            grid: Threadgroup grid dimensions.
            threadgroup: Threads per threadgroup.
            buffers: Metal buffers to bind.

        Raises:
            RuntimeError: If expert doesn't have an active command buffer.
        """
        if expert_idx not in self._active_jobs:
            raise RuntimeError(f"No active command buffer for expert {expert_idx}. "
                             "Call begin_expert_command() first.")

        job = self._active_jobs[expert_idx]
        job.kernel_name = kernel_name
        job.grid = grid
        job.threadgroup = threadgroup
        job.buffers = buffers

        pipeline = self._get_pipeline(kernel_name)
        job.encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffers):
            job.encoder.setBuffer_offset_atIndex_(buf, 0, i)

        job.encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )

    def commit_all_async(self) -> ExpertExecutionBarrier:
        """Commit all command buffers for asynchronous execution.

        All expert command buffers are committed simultaneously, allowing
        the GPU to schedule them in parallel. Returns a barrier that can
        be used to wait for completion without blocking the commit path.
        """
        if self._committed:
            raise RuntimeError("Batch already committed.")
        if not self._active_jobs:
            raise RuntimeError("No active jobs to commit.")

        # End encoding and commit all command buffers
        barrier = ExpertExecutionBarrier()
        for job in self._active_jobs.values():
            job.encoder.endEncoding()
            job.command_buffer.commit()
            barrier.track(job.command_buffer, expert_idx=job.expert_idx)

        # Move active jobs to jobs dict
        self._jobs = self._active_jobs
        self._active_jobs = {}
        self._committed = True
        return barrier

    def wait_for_all(self) -> None:
        """Block until all committed command buffers complete."""
        if not self._committed:
            raise RuntimeError("Batch not committed. Call commit_all_async() first.")

        for job in self._jobs.values():
            job.command_buffer.waitUntilCompleted()

        self._jobs.clear()
        self._committed = False

    def wait_for_expert(self, expert_idx: int) -> bool:
        """Wait for a specific expert's command buffer to complete.

        Args:
            expert_idx: Index of the expert to wait for.

        Returns:
            True if found and waited, False if expert not in batch.
        """
        if expert_idx in self._jobs:
            self._jobs[expert_idx].command_buffer.waitUntilCompleted()
            return True
        return False

    def reset(self) -> None:
        """Reset the executor for a new batch."""
        self._jobs.clear()
        self._active_jobs.clear()
        self._committed = False

    def batch_dispatch(self) -> _ExpertBatchContext:
        """Return a context manager for batch dispatch.

        Usage:
            with executor.batch_dispatch() as batch:
                batch.dispatch_expert_async(...)
                batch.dispatch_expert_async(...)
            # Automatically commits and waits
        """
        return _ExpertBatchContext(self)


class _ExpertBatchContext:
    """Context manager for batched expert dispatch."""

    __slots__ = ("_executor", "_expert_count")

    def __init__(self, executor: ParallelExpertExecutor) -> None:
        self._executor = executor
        self._expert_count = 0

    def __enter__(self) -> _ExpertBatchContext:
        self._executor.begin_batch()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self._executor.commit_all_async()
            self._executor.wait_for_all()
        self._executor.reset()

    def dispatch_expert_async(
        self,
        expert_idx: int,
        kernel: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
    ) -> None:
        """Dispatch an expert asynchronously as part of the batch.

        Args:
            expert_idx: Index of the expert.
            kernel: Kernel function name.
            grid: Threadgroup grid.
            threadgroup: Threads per threadgroup.
            buffers: Metal buffers.
        """
        self._executor.begin_expert_command(expert_idx)
        self._executor.encode_expert_kernel(expert_idx, kernel, grid, threadgroup, buffers)
        self._expert_count += 1


# -----------------------------------------------------------------------------
# High-Level MoE Parallel Execution Functions
# -----------------------------------------------------------------------------


def execute_experts_parallel(
    lib: Any,
    expert_configs: Sequence[tuple[int, str, tuple[int, int, int], tuple[int, int, int], Sequence[Any]]],
) -> None:
    """Execute multiple expert kernels in parallel.

    This is a convenience function that creates a ParallelExpertExecutor,
    encodes all dispatches, commits them simultaneously, and waits.

    Args:
        lib: MetalKernelLibrary instance.
        expert_configs: Sequence of (expert_idx, kernel_name, grid, threadgroup, buffers).
            Each tuple defines one expert kernel dispatch.

    Example:
        expert_configs = [
            (0, "expert_gemm", (16, 1, 1), (128, 1, 1), [buf0, w0, out0]),
            (1, "expert_gemm", (16, 1, 1), (128, 1, 1), [buf1, w1, out1]),
            (2, "expert_gemm", (8, 1, 1), (128, 1, 1), [buf2, w2, out2]),
        ]
        execute_experts_parallel(lib, expert_configs)
    """
    executor = ParallelExpertExecutor(lib)

    with executor.batch_dispatch() as batch:
        for expert_idx, kernel, grid, threadgroup, buffers in expert_configs:
            batch.dispatch_expert_async(expert_idx, kernel, grid, threadgroup, buffers)


def execute_experts_parallel_async(
    lib: Any,
    expert_configs: Sequence[tuple[int, str, tuple[int, int, int], tuple[int, int, int], Sequence[Any]]],
) -> ExpertExecutionBarrier:
    """Submit multiple expert kernels in parallel without blocking.

    Args:
        lib: MetalKernelLibrary instance.
        expert_configs: Sequence of (expert_idx, kernel_name, grid, threadgroup, buffers).

    Returns:
        ExpertExecutionBarrier for optional synchronization.
    """
    executor = ParallelExpertExecutor(lib)
    executor.begin_batch()
    for expert_idx, kernel, grid, threadgroup, buffers in expert_configs:
        executor.begin_expert_command(expert_idx)
        executor.encode_expert_kernel(expert_idx, kernel, grid, threadgroup, buffers)
    return executor.commit_all_async()


@dataclass
class ExpertWorkItem:
    """Work item for async expert execution with PyTorch tensors.

    Attributes:
        expert_idx: Index of the expert.
        input_tensor: Input activation tensor [num_tokens, hidden_dim].
        weight_tensor: Weight tensor for the expert.
        output_tensor: Pre-allocated output tensor.
        kernel_name: Name of the Metal kernel to use.
    """

    expert_idx: int
    input_tensor: torch.Tensor
    weight_tensor: torch.Tensor
    output_tensor: torch.Tensor
    kernel_name: str = "moe_expert_gemm_fp4"


class AsyncMoEExecutor:
    """High-level async executor for MoE with PyTorch tensor integration.

    This class provides a PyTorch-friendly interface for async expert execution,
    handling the tensor-to-buffer conversion and buffer management.

    Supports both multi-expert (parallel) and single-expert (sparse) execution paths.

    Usage:
        executor = AsyncMoEExecutor()

        # Multi-expert execution (tokens activate multiple experts)
        work_items = [
            ExpertWorkItem(0, tokens_0, weight_0, output_0),
            ExpertWorkItem(1, tokens_1, weight_1, output_1),
        ]
        executor.execute_parallel(work_items)

        # Sparse single-expert execution (each token activates only one expert)
        executor.execute_sparse_experts(
            activations=activations,
            expert_weights=expert_weights,
            token_to_expert_map=token_to_expert_map,
        )
    """

    __slots__ = ("_lib", "_executor", "_buffer_cache")

    def __init__(self) -> None:
        """Initialize async MoE executor."""
        require_metal()
        self._lib = get_default_library()
        self._executor = ParallelExpertExecutor(self._lib)
        self._buffer_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _get_or_create_buffer(self, tensor: torch.Tensor) -> Any:
        """Get or create Metal buffer for a tensor."""
        # Check cache first
        if tensor in self._buffer_cache:
            return self._buffer_cache[tensor]

        # Create new buffer
        buffer = mps_tensor_to_metal_buffer(tensor, self._lib.device)
        self._buffer_cache[tensor] = buffer
        return buffer

    def execute_parallel(self, work_items: Sequence[ExpertWorkItem]) -> None:
        """Execute multiple expert work items in parallel.

        Args:
            work_items: Sequence of ExpertWorkItem defining the work for each expert.
        """
        require_mps()

        with self._executor.batch_dispatch() as batch:
            for item in work_items:
                # Convert tensors to Metal buffers
                input_buf = self._get_or_create_buffer(item.input_tensor)
                weight_buf = self._get_or_create_buffer(item.weight_tensor)
                output_buf = self._get_or_create_buffer(item.output_tensor)

                # Calculate grid dimensions based on work size
                num_tokens = item.input_tensor.shape[0]
                threads_per_tg = 128
                num_threadgroups = (num_tokens + threads_per_tg - 1) // threads_per_tg

                batch.dispatch_expert_async(
                    expert_idx=item.expert_idx,
                    kernel=item.kernel_name,
                    grid=(num_threadgroups, 1, 1),
                    threadgroup=(threads_per_tg, 1, 1),
                    buffers=[input_buf, weight_buf, output_buf],
                )

    def execute_parallel_with_offsets(
        self,
        activations: torch.Tensor,
        expert_weights: Sequence[torch.Tensor],
        expert_offsets: torch.Tensor,
        kernel_name: str = "moe_expert_gemm_fp4",
    ) -> torch.Tensor:
        """Execute experts in parallel using offset-based dispatch.

        Args:
            activations: [total_tokens, hidden_dim] grouped activations.
            expert_weights: List of weight tensors, one per expert.
            expert_offsets: [num_experts + 1] offset array defining token ranges.
            kernel_name: Metal kernel to use.

        Returns:
            [total_tokens, hidden_dim] expert outputs.
        """
        require_mps()

        num_experts = len(expert_weights)
        device = activations.device
        hidden_dim = activations.shape[1]

        # Pre-allocate output tensor
        output = torch.empty_like(activations)

        # Build work items from offsets
        work_items = []
        for expert_idx in range(num_experts):
            start = int(expert_offsets[expert_idx].item())
            end = int(expert_offsets[expert_idx + 1].item())

            if start < end:
                expert_input = activations[start:end]
                expert_output = output[start:end]

                work_items.append(ExpertWorkItem(
                    expert_idx=expert_idx,
                    input_tensor=expert_input,
                    weight_tensor=expert_weights[expert_idx],
                    output_tensor=expert_output,
                    kernel_name=kernel_name,
                ))

        self.execute_parallel(work_items)
        return output

    def execute_sparse_experts(
        self,
        activations: torch.Tensor,
        expert_weights: Sequence[torch.Tensor],
        token_to_expert_map: torch.Tensor,
        kernel_name: str = "moe_expert_gemm_fp4",
    ) -> torch.Tensor:
        """Execute experts in sparse mode where each token activates only one expert.

        This is an optimized path for tokens that route to exactly one expert,
        avoiding the overhead of parallel coordination when unnecessary.

        Args:
            activations: [total_tokens, hidden_dim] input activations.
            expert_weights: List of weight tensors, one per expert.
            token_to_expert_map: [total_tokens] tensor mapping each token to its expert index.
            kernel_name: Metal kernel to use.

        Returns:
            [total_tokens, hidden_dim] expert outputs.

        Example:
            # Each token routes to exactly one expert
            token_to_expert_map = torch.tensor([0, 2, 1, 0, 2])  # 5 tokens, 3 experts
            output = executor.execute_sparse_experts(
                activations,
                expert_weights,
                token_to_expert_map,
            )
        """
        require_mps()

        device = activations.device
        hidden_dim = activations.shape[1]

        # Pre-allocate output tensor
        output = torch.empty_like(activations)

        # Group tokens by their assigned expert
        expert_indices: dict[int, list[int]] = {i: [] for i in range(len(expert_weights))}
        for token_idx, expert_idx in enumerate(token_to_expert_map.cpu().tolist()):
            expert_indices[int(expert_idx)].append(token_idx)

        # Build work items only for experts that have tokens
        work_items = []
        for expert_idx, token_indices in expert_indices.items():
            if not token_indices:
                continue

            # Gather tokens for this expert
            token_tensor = torch.tensor(token_indices, device=device, dtype=torch.long)
            expert_input = activations[token_tensor]
            expert_output = output[token_tensor]

            work_items.append(ExpertWorkItem(
                expert_idx=expert_idx,
                input_tensor=expert_input,
                weight_tensor=expert_weights[expert_idx],
                output_tensor=expert_output,
                kernel_name=kernel_name,
            ))

        if work_items:
            self.execute_parallel(work_items)

        return output

    def execute_sparse_single_expert(
        self,
        activations: torch.Tensor,
        expert_weights: Sequence[torch.Tensor],
        token_to_expert_map: torch.Tensor,
        kernel_name: str = "moe_expert_gemm_fp4",
    ) -> torch.Tensor:
        """Fast path for sparse execution when each token uses exactly one expert.
        
        This method optimizes for the common case where tokens route to single experts,
        avoiding the overhead of parallel dispatch coordination. It processes experts
        sequentially but with minimal overhead.
        
        Args:
            activations: [total_tokens, hidden_dim] input activations.
            expert_weights: List of weight tensors, one per expert.
            token_to_expert_map: [total_tokens] tensor mapping each token to its expert index.
            kernel_name: Metal kernel to use.
            
        Returns:
            [total_tokens, hidden_dim] expert outputs.
            
        Example:
            # Sparse routing: each token → one expert
            token_to_expert_map = torch.tensor([0, 2, 1, 0, 2])
            output = executor.execute_sparse_single_expert(
                activations,
                expert_weights,
                token_to_expert_map,
            )
        """
        require_mps()
        
        device = activations.device
        hidden_dim = activations.shape[1]
        output = torch.empty_like(activations)
        
        # Group tokens by expert
        expert_to_tokens: dict[int, list[int]] = {}
        for token_idx, expert_idx in enumerate(token_to_expert_map.cpu().tolist()):
            expert_idx = int(expert_idx)
            if expert_idx not in expert_to_tokens:
                expert_to_tokens[expert_idx] = []
            expert_to_tokens[expert_idx].append(token_idx)
        
        # Process each expert sequentially (fast path, minimal overhead)
        for expert_idx, token_indices in expert_to_tokens.items():
            if not token_indices:
                continue
                
            token_tensor = torch.tensor(token_indices, device=device, dtype=torch.long)
            expert_input = activations[token_tensor]
            expert_output = output[token_tensor]
            
            # Single expert dispatch (no batch overhead)
            input_buf = self._get_or_create_buffer(expert_input)
            weight_buf = self._get_or_create_buffer(expert_weights[expert_idx])
            output_buf = self._get_or_create_buffer(expert_output)
            
            num_tokens = expert_input.shape[0]
            threads_per_tg = 128
            num_threadgroups = (num_tokens + threads_per_tg - 1) // threads_per_tg
            
            # Direct dispatch without parallel coordination
            async_buf = AsyncExpertCommandBuffer(self._lib)
            async_buf.begin_encoding()
            async_buf.encode_kernel(
                kernel_name,
                (num_threadgroups, 1, 1),
                (threads_per_tg, 1, 1),
                [input_buf, weight_buf, output_buf],
            )
            async_buf.commit_async()
            async_buf.wait_for_completion()
        
        return output


# -----------------------------------------------------------------------------
# Synchronization Primitives
# -----------------------------------------------------------------------------


class ExpertExecutionBarrier:
    """Barrier for synchronizing multiple async expert executions.

    This class allows fine-grained synchronization control when executing
    experts in parallel. It supports waiting for subsets of experts.

    Usage:
        barrier = ExpertExecutionBarrier()

        # Submit work
        for i in range(num_experts):
            cmd_buf = create_command_buffer_for_expert(i)
            barrier.track(cmd_buf, expert_idx=i)
            cmd_buf.commit()

        # Wait for specific experts
        barrier.wait_for([0, 1, 2])

        # Or wait for all
        barrier.wait_for_all()
    """

    __slots__ = ("_command_buffers", "_expert_map")

    def __init__(self) -> None:
        """Initialize execution barrier."""
        self._command_buffers: list[Any] = []
        self._expert_map: dict[int, Any] = {}

    def track(self, command_buffer: Any, expert_idx: int | None = None) -> None:
        """Track a command buffer for synchronization.

        Args:
            command_buffer: Metal command buffer to track.
            expert_idx: Optional expert index for selective waiting.
        """
        self._command_buffers.append(command_buffer)
        if expert_idx is not None:
            self._expert_map[expert_idx] = command_buffer

    def wait_for_all(self) -> None:
        """Wait for all tracked command buffers to complete."""
        for cmd_buf in self._command_buffers:
            cmd_buf.waitUntilCompleted()

    def wait_for(self, expert_indices: Sequence[int]) -> None:
        """Wait for specific experts to complete.

        Args:
            expert_indices: Indices of experts to wait for.
        """
        for idx in expert_indices:
            if idx in self._expert_map:
                self._expert_map[idx].waitUntilCompleted()

    def clear(self) -> None:
        """Clear all tracked command buffers."""
        self._command_buffers.clear()
        self._expert_map.clear()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def create_expert_constant_buffer(lib: Any, value: int | float, dtype: np.dtype) -> Any:
    """Create a constant buffer for kernel arguments.

    Args:
        lib: MetalKernelLibrary instance.
        value: Scalar value to encode.
        dtype: NumPy dtype for the value.

    Returns:
        Metal buffer containing the constant value.
    """
    arr = np.array([value], dtype=dtype)
    return lib.device.newBufferWithBytes_length_options_(
        arr.tobytes(),
        arr.nbytes,
        Metal.MTLResourceStorageModeShared,
    )


def dispatch_single_expert_async(
    lib: Any,
    kernel_name: str,
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    buffers: Sequence[Any],
) -> AsyncExpertCommandBuffer:
    """Dispatch a single expert asynchronously and return the command buffer.

    This is a convenience function for simple async dispatch of a single expert.

    Args:
        lib: MetalKernelLibrary instance.
        kernel_name: Name of the kernel function.
        grid: Threadgroup grid.
        threadgroup: Threads per threadgroup.
        buffers: Metal buffers.

    Returns:
        AsyncExpertCommandBuffer that can be waited on.
    """
    async_buf = AsyncExpertCommandBuffer(lib)
    async_buf.begin_encoding()
    async_buf.encode_kernel(kernel_name, grid, threadgroup, buffers)
    async_buf.commit_async()
    return async_buf


__all__ = [
    "AsyncExpertCommandBuffer",
    "AsyncMoEExecutor",
    "ExpertDispatchJob",
    "ExpertExecutionBarrier",
    "ExpertWorkItem",
    "ParallelExpertExecutor",
    "create_expert_constant_buffer",
    "dispatch_single_expert_async",
    "execute_experts_parallel",
    "execute_experts_parallel_async",
]
