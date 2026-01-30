"""
Metal-accelerated token sampling for autoregressive generation.

Provides GPU-accelerated sampling strategies using Metal compute kernels:
    - Greedy decoding (argmax)
    - Temperature scaling
    - Top-k sampling
    - Top-p (nucleus) sampling
    - Repetition penalty

Works with PyTorch MPS tensors, dispatching Metal kernels directly via PyObjC.

Usage:
    from metal_marlin.sampler import MetalSampler

    sampler = MetalSampler(vocab_size=32000)

    # Greedy decoding
    token_id = sampler.argmax(logits)

    # Temperature + top-p sampling
    token_id = sampler.sample(logits, temperature=0.7, top_p=0.9)

    # With repetition penalty
    token_id = sampler.sample(
        logits, temperature=0.7, top_p=0.9,
        generated_ids=[1, 2, 3], repetition_penalty=1.1
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._compat import Metal, torch
from .metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)

if TYPE_CHECKING:
    import torch as torch_typing


@dataclass
class SamplingConfig:
    """Configuration for token sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: bool = True


class MetalSampler:
    """
    Metal-accelerated token sampler.

    Dispatches sampling kernels via PyObjC Metal bindings, operating on
    PyTorch MPS tensors without copying to CPU.

    Attributes:
        vocab_size: Size of the vocabulary
        lib: MetalKernelLibrary containing compiled sampling kernels
    """

    def __init__(
        self,
        vocab_size: int,
        lib: MetalKernelLibrary | None = None,
        seed: int | None = None,
    ):
        """Initialize the Metal sampler.

        Args:
            vocab_size: Size of the vocabulary
            lib: Optional MetalKernelLibrary. If None, uses the default library.
            seed: Random seed for sampling. If None, uses current time.
        """
        require_metal()
        require_mps()

        self.vocab_size = vocab_size
        self.lib = lib if lib is not None else get_default_library()
        self._seed = seed if seed is not None else int(np.random.randint(0, 2**63))

        # Pre-create output buffers for single-token sampling (batch_size=1)
        self._output_buffer = self.lib.device.newBufferWithLength_options_(
            4,  # uint32
            Metal.MTLResourceStorageModeShared,
        )

    def _get_seed(self) -> int:
        """Get and increment the random seed."""
        seed = self._seed
        self._seed = (self._seed + 1) % (2**63)
        return seed

    def argmax(self, logits: torch_typing.Tensor) -> int:
        """Greedy decoding: return the token with highest logit.

        Args:
            logits: Logits tensor [1, vocab_size] or [vocab_size], fp16/fp32, on MPS

        Returns:
            Token ID with maximum logit value
        """
        require_mps()

        # Ensure correct shape
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            # [1, 1, vocab_size] -> [1, vocab_size]
            logits = logits.squeeze(1)

        batch_size = logits.shape[0]
        vocab = logits.shape[-1]

        if vocab != self.vocab_size:
            raise ValueError(f"Expected vocab_size {self.vocab_size}, got {vocab}")

        # Create Metal buffers
        device = self.lib.device
        logits_buf = mps_tensor_to_metal_buffer(logits.contiguous(), device)

        # Output buffer
        output_buf = device.newBufferWithLength_options_(
            batch_size * 4, Metal.MTLResourceStorageModeShared
        )

        # Params buffer
        params = np.array([vocab, batch_size], dtype=np.uint32)
        params_buf = device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        # Select kernel based on dtype
        kernel_name = "argmax_fp16" if logits.dtype == torch.float16 else "argmax"

        # Dispatch
        dispatch_kernel(
            self.lib,
            function_name=kernel_name,
            grid=(batch_size, 1, 1),
            threadgroup=(min(256, vocab), 1, 1),
            buffers=[logits_buf, output_buf, params_buf],
            wait=True,
        )

        # Read result
        result = np.frombuffer(output_buf.contents().as_buffer(batch_size * 4), dtype=np.uint32)
        return int(result[0])

    def sample_categorical(
        self,
        logits: torch_typing.Tensor,
        temperature: float = 1.0,
    ) -> int:
        """Sample from the categorical distribution defined by logits.

        Uses the Gumbel-max trick for efficient GPU sampling.

        Args:
            logits: Logits tensor [1, vocab_size] or [vocab_size], fp16/fp32, on MPS
            temperature: Temperature for scaling logits

        Returns:
            Sampled token ID
        """
        require_mps()

        # Ensure correct shape
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.squeeze(1)

        batch_size = logits.shape[0]
        vocab = logits.shape[-1]

        # Apply temperature if needed
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # Create Metal buffers
        device = self.lib.device
        logits_buf = mps_tensor_to_metal_buffer(logits.contiguous(), device)

        output_buf = device.newBufferWithLength_options_(
            batch_size * 4, Metal.MTLResourceStorageModeShared
        )

        # Repack as the struct expects uint, uint, ulong
        params_packed = np.zeros(4, dtype=np.uint32)
        params_packed[0] = vocab
        params_packed[1] = batch_size
        params_packed[2:4].view(np.uint64)[0] = self._get_seed()
        params_buf = device.newBufferWithBytes_length_options_(
            params_packed.tobytes(), params_packed.nbytes, Metal.MTLResourceStorageModeShared
        )

        kernel_name = (
            "sample_categorical_logits_fp16"
            if logits.dtype == torch.float16
            else "sample_categorical_logits"
        )

        dispatch_kernel(
            self.lib,
            function_name=kernel_name,
            grid=(batch_size, 1, 1),
            threadgroup=(min(256, vocab), 1, 1),
            buffers=[logits_buf, output_buf, params_buf],
            wait=True,
        )

        result = np.frombuffer(output_buf.contents().as_buffer(batch_size * 4), dtype=np.uint32)
        return int(result[0])

    def sample_top_k(
        self,
        logits: torch_typing.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> int:
        """Sample from the top-k tokens.

        Args:
            logits: Logits tensor [1, vocab_size], fp16/fp32, on MPS
            k: Number of top tokens to consider
            temperature: Temperature for scaling logits

        Returns:
            Sampled token ID
        """
        require_mps()

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.squeeze(1)

        batch_size = logits.shape[0]
        vocab = logits.shape[-1]

        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        device = self.lib.device
        logits_buf = mps_tensor_to_metal_buffer(logits.contiguous(), device)

        output_buf = device.newBufferWithLength_options_(
            batch_size * 4, Metal.MTLResourceStorageModeShared
        )

        # Params: vocab_size, batch_size, k, seed
        params_packed = np.zeros(6, dtype=np.uint32)
        params_packed[0] = vocab
        params_packed[1] = batch_size
        params_packed[2] = min(k, vocab)
        params_packed[3] = 0  # padding
        params_packed[4:6].view(np.uint64)[0] = self._get_seed()
        params_buf = device.newBufferWithBytes_length_options_(
            params_packed.tobytes(), params_packed.nbytes, Metal.MTLResourceStorageModeShared
        )

        dispatch_kernel(
            self.lib,
            function_name="sample_top_k",
            grid=(batch_size, 1, 1),
            threadgroup=(min(128, vocab), 1, 1),  # Smaller TG for heap operations
            buffers=[logits_buf, output_buf, params_buf],
            wait=True,
        )

        result = np.frombuffer(output_buf.contents().as_buffer(batch_size * 4), dtype=np.uint32)
        return int(result[0])

    def sample_top_p(
        self,
        logits: torch_typing.Tensor,
        p: float,
        temperature: float = 1.0,
    ) -> int:
        """Nucleus (top-p) sampling.

        Sample from the smallest set of tokens whose cumulative probability
        exceeds threshold p.

        Args:
            logits: Logits tensor [1, vocab_size], fp16/fp32, on MPS
            p: Cumulative probability threshold (0 < p <= 1)
            temperature: Temperature for scaling logits

        Returns:
            Sampled token ID
        """
        require_mps()

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.squeeze(1)

        batch_size = logits.shape[0]
        vocab = logits.shape[-1]

        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        device = self.lib.device
        logits_buf = mps_tensor_to_metal_buffer(logits.float().contiguous(), device)

        output_buf = device.newBufferWithLength_options_(
            batch_size * 4, Metal.MTLResourceStorageModeShared
        )

        # Workspace for softmax probs and sorting
        workspace_size = batch_size * vocab * 2 * 4  # 2x for probs + indices, 4 bytes each
        workspace_buf = device.newBufferWithLength_options_(
            workspace_size, Metal.MTLResourceStorageModeShared
        )

        # Params: vocab_size, batch_size, p, seed
        # Struct: uint, uint, float, ulong
        params = np.zeros(5, dtype=np.uint32)
        params[0] = vocab
        params[1] = batch_size
        params[2] = np.float32(p).view(np.uint32)
        params[3] = 0  # padding
        params[3:5].view(np.uint64)[0] = self._get_seed()
        params_buf = device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        dispatch_kernel(
            self.lib,
            function_name="sample_top_p",
            grid=(batch_size, 1, 1),
            threadgroup=(min(256, vocab), 1, 1),
            buffers=[logits_buf, output_buf, workspace_buf, params_buf],
            wait=True,
        )

        result = np.frombuffer(output_buf.contents().as_buffer(batch_size * 4), dtype=np.uint32)
        return int(result[0])

    def apply_repetition_penalty(
        self,
        logits: torch_typing.Tensor,
        generated_ids: list[int],
        penalty: float,
    ) -> torch_typing.Tensor:
        """Apply repetition penalty to logits in-place.

        For each token in generated_ids:
            if logits[token] > 0: logits[token] /= penalty
            else: logits[token] *= penalty

        Args:
            logits: Logits tensor [1, vocab_size], fp16/fp32, on MPS (modified in-place)
            generated_ids: List of previously generated token IDs
            penalty: Repetition penalty (1.0 = no penalty)

        Returns:
            Modified logits tensor (same object, modified in-place)
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        require_mps()

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.squeeze(1)

        batch_size = logits.shape[0]
        vocab = logits.shape[-1]

        device = self.lib.device

        # Ensure logits are contiguous for in-place modification
        if not logits.is_contiguous():
            logits = logits.contiguous()

        logits_buf = mps_tensor_to_metal_buffer(logits, device)

        # Create generated_ids buffer
        gen_ids = np.array(generated_ids, dtype=np.uint32)
        gen_ids_buf = device.newBufferWithBytes_length_options_(
            gen_ids.tobytes(), gen_ids.nbytes, Metal.MTLResourceStorageModeShared
        )

        # Params: vocab_size, batch_size, num_generated, penalty
        params = np.zeros(4, dtype=np.uint32)
        params[0] = vocab
        params[1] = batch_size
        params[2] = len(generated_ids)
        params[3] = np.float32(penalty).view(np.uint32)
        params_buf = device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        kernel_name = (
            "apply_repetition_penalty_fp16"
            if logits.dtype == torch.float16
            else "apply_repetition_penalty"
        )

        dispatch_kernel(
            self.lib,
            function_name=kernel_name,
            grid=(batch_size, 1, 1),
            threadgroup=(min(256, len(generated_ids)), 1, 1),
            buffers=[logits_buf, gen_ids_buf, params_buf],
            wait=True,
        )

        return logits

    def sample(
        self,
        logits: torch_typing.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        generated_ids: list[int] | None = None,
    ) -> int:
        """Sample a token using the specified strategy.

        Applies sampling modifiers in order:
            1. Repetition penalty (if enabled)
            2. Temperature scaling (if != 1.0)
            3. Top-k filtering (if > 0)
            4. Top-p filtering (if < 1.0)
            5. Categorical sampling

        Args:
            logits: Logits tensor [1, vocab_size], fp16/fp32, on MPS
            temperature: Temperature for scaling (higher = more random)
            top_p: Nucleus sampling threshold (0 < p <= 1)
            top_k: Top-k filtering (0 = disabled)
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            generated_ids: Previously generated token IDs for repetition penalty

        Returns:
            Sampled token ID
        """
        require_mps()

        # Make a copy to avoid modifying the original
        logits = logits.clone()

        # Apply repetition penalty if needed
        if repetition_penalty != 1.0 and generated_ids:
            logits = self.apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        # Choose sampling strategy
        if temperature == 0:
            # Greedy decoding
            return self.argmax(logits)

        if top_k > 0:
            # Top-k sampling
            return self.sample_top_k(logits, top_k, temperature)

        if top_p < 1.0:
            # Top-p sampling
            return self.sample_top_p(logits, top_p, temperature)

        # Pure temperature sampling
        return self.sample_categorical(logits, temperature)


# Convenience function
def sample_next_token(
    logits: torch_typing.Tensor,
    config: SamplingConfig | None = None,
    generated_ids: list[int] | None = None,
    sampler: MetalSampler | None = None,
) -> int:
    """Sample the next token from logits.

    Args:
        logits: Logits tensor [1, vocab_size], on MPS
        config: Sampling configuration. If None, uses defaults.
        generated_ids: Previously generated token IDs for repetition penalty.
        sampler: MetalSampler instance. If None, creates one.

    Returns:
        Sampled token ID
    """
    if config is None:
        config = SamplingConfig()

    vocab_size = logits.shape[-1]

    if sampler is None:
        sampler = MetalSampler(vocab_size=vocab_size)

    if not config.do_sample:
        return sampler.argmax(logits)

    return sampler.sample(
        logits,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        generated_ids=generated_ids,
    )
