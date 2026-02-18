"""Validation utilities for Paged Attention vs Linear Attention parity.

This module provides comprehensive validation tools to ensure that paged attention
implementations produce numerically equivalent results to standard linear attention
within acceptable tolerance bounds.

Usage:
    from metal_marlin.paged.validation import validate_paged_linear_parity
    
    # Validate parity for a specific configuration
    result = validate_paged_linear_parity(
        query=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
    )
    
    if result.is_valid:
        print(f"Parity validated: max_diff={result.max_diff:.6f}")
    else:
        print(f"Parity check failed: {result.error_message}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


@dataclass
class ParityValidationResult:
    """Result of parity validation between paged and linear attention.
    
    Attributes:
        is_valid: Whether the outputs are within tolerance
        max_diff: Maximum absolute difference between outputs
        mean_diff: Mean absolute difference between outputs
        rel_diff: Maximum relative difference between outputs
        paged_output: Output from paged attention
        linear_output: Output from linear attention
        atol: Absolute tolerance used for comparison
        rtol: Relative tolerance used for comparison
        error_message: Detailed error message if validation failed
    """
    is_valid: bool
    max_diff: float
    mean_diff: float
    rel_diff: float
    paged_output: NDArray[Any] | None = None
    linear_output: NDArray[Any] | None = None
    atol: float = 1e-5
    rtol: float = 0.01
    error_message: str | None = None
    
    def __str__(self) -> str:
        if self.is_valid:
            return (
                f"ParityValidationResult(valid=True, "
                f"max_diff={self.max_diff:.6e}, "
                f"mean_diff={self.mean_diff:.6e}, "
                f"rel_diff={self.rel_diff:.6e})"
            )
        else:
            return (
                f"ParityValidationResult(valid=False, "
                f"max_diff={self.max_diff:.6e}, "
                f"error={self.error_message})"
            )


@dataclass
class ValidationConfig:
    """Configuration for parity validation.
    
    Attributes:
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for Inf values
        require_exact_shape: Whether shapes must match exactly
        verbose: Whether to print detailed information
    """
    atol: float = 1e-5
    rtol: float = 0.01
    check_nan: bool = True
    check_inf: bool = True
    require_exact_shape: bool = True
    verbose: bool = False


def _gather_kv_from_blocks(
    k_cache: NDArray[Any],
    v_cache: NDArray[Any],
    block_tables: NDArray[Any],
    context_lens: NDArray[Any],
    block_size: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Gather K/V from paged blocks into contiguous arrays.
    
    This converts paged KV cache layout to standard linear attention layout
    for comparison purposes.
    
    Args:
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: Block indices [num_seqs, max_blocks_per_seq]
        context_lens: Context lengths [num_seqs]
        block_size: Tokens per block
        
    Returns:
        Tuple of (keys, values) in linear layout
        [num_seqs, max_context_len, num_kv_heads, head_dim]
    """
    num_seqs = block_tables.shape[0]
    max_blocks = block_tables.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    max_context_len = max_blocks * block_size
    
    k_linear = np.zeros((num_seqs, max_context_len, num_kv_heads, head_dim), 
                        dtype=k_cache.dtype)
    v_linear = np.zeros((num_seqs, max_context_len, num_kv_heads, head_dim),
                        dtype=v_cache.dtype)
    
    for seq_idx in range(num_seqs):
        ctx_len = int(context_lens[seq_idx])
        seq_keys = []
        seq_values = []
        
        for blk_idx in range((ctx_len + block_size - 1) // block_size):
            phys_block = int(block_tables[seq_idx, blk_idx])
            seq_keys.append(k_cache[phys_block])
            seq_values.append(v_cache[phys_block])
        
        if seq_keys:
            k_concat = np.concatenate(seq_keys, axis=0)[:ctx_len]
            v_concat = np.concatenate(seq_values, axis=0)[:ctx_len]
            k_linear[seq_idx, :ctx_len] = k_concat
            v_linear[seq_idx, :ctx_len] = v_concat
    
    return k_linear, v_linear


def compute_linear_attention(
    query: NDArray[Any],
    keys: NDArray[Any],
    values: NDArray[Any],
    scale: float,
    is_causal: bool = False,
) -> NDArray[Any]:
    """Compute standard linear (scaled dot-product) attention.
    
    Args:
        query: Query tensor [num_seqs, num_heads, q_len, head_dim]
        keys: Key tensor [num_seqs, num_kv_heads, kv_len, head_dim]
        values: Value tensor [num_seqs, num_kv_heads, kv_len, head_dim]
        scale: Attention scale factor
        is_causal: Whether to apply causal masking
        
    Returns:
        Attention output [num_seqs, num_heads, q_len, head_dim]
    """
    from scipy.special import softmax
    
    num_seqs, num_heads, q_len, head_dim = query.shape
    num_kv_heads = keys.shape[1]
    kv_len = keys.shape[2]
    
    # Handle GQA by repeating KV heads
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        keys = np.repeat(keys, repeat_factor, axis=1)
        values = np.repeat(values, repeat_factor, axis=1)
    
    # Compute attention scores
    # [num_seqs, num_heads, q_len, head_dim] @ [num_seqs, num_heads, head_dim, kv_len]
    # = [num_seqs, num_heads, q_len, kv_len]
    scores = np.matmul(query, np.transpose(keys, (0, 1, 3, 2))) * scale
    
    # Apply causal mask if needed
    if is_causal:
        # Create causal mask: positions i can only attend to positions <= i
        q_pos = np.arange(q_len)[:, None]  # [q_len, 1]
        kv_pos = np.arange(kv_len)[None, :]  # [1, kv_len]
        causal_mask = q_pos >= kv_pos  # [q_len, kv_len]
        
        # Expand for batch and heads
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, q_len, kv_len]
        scores = np.where(causal_mask, scores, float("-inf"))
    
    # Softmax and apply to values
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, values)  # [num_seqs, num_heads, q_len, head_dim]
    
    return output


def validate_paged_linear_parity(
    paged_output: NDArray[Any],
    linear_output: NDArray[Any],
    config: ValidationConfig | None = None,
) -> ParityValidationResult:
    """Validate parity between paged and linear attention outputs.
    
    Args:
        paged_output: Output from paged attention
        linear_output: Output from linear attention
        config: Validation configuration
        
    Returns:
        ParityValidationResult with detailed comparison information
    """
    if config is None:
        config = ValidationConfig()
    
    # Check shapes
    if config.require_exact_shape and paged_output.shape != linear_output.shape:
        return ParityValidationResult(
            is_valid=False,
            max_diff=float("inf"),
            mean_diff=float("inf"),
            rel_diff=float("inf"),
            error_message=(
                f"Shape mismatch: paged={paged_output.shape}, "
                f"linear={linear_output.shape}"
            ),
            atol=config.atol,
            rtol=config.rtol,
        )
    
    # Check for NaN/Inf
    if config.check_nan:
        paged_nan = np.isnan(paged_output).any()
        linear_nan = np.isnan(linear_output).any()
        if paged_nan or linear_nan:
            return ParityValidationResult(
                is_valid=False,
                max_diff=float("nan"),
                mean_diff=float("nan"),
                rel_diff=float("nan"),
                error_message=f"NaN detected: paged={paged_nan}, linear={linear_nan}",
                atol=config.atol,
                rtol=config.rtol,
            )
    
    if config.check_inf:
        paged_inf = np.isinf(paged_output).any()
        linear_inf = np.isinf(linear_output).any()
        if paged_inf or linear_inf:
            return ParityValidationResult(
                is_valid=False,
                max_diff=float("inf"),
                mean_diff=float("inf"),
                rel_diff=float("inf"),
                error_message=f"Inf detected: paged={paged_inf}, linear={linear_inf}",
                atol=config.atol,
                rtol=config.rtol,
            )
    
    # Compute differences
    abs_diff = np.abs(paged_output - linear_output)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    
    # Compute relative difference (avoiding division by zero)
    abs_linear = np.abs(linear_output)
    rel_diff = np.where(
        abs_linear > 1e-10,
        abs_diff / abs_linear,
        abs_diff,
    )
    max_rel_diff = float(np.max(rel_diff))
    
    # Check tolerance
    is_valid = (max_diff <= config.atol) or (max_rel_diff <= config.rtol)
    
    error_message = None
    if not is_valid:
        error_message = (
            f"Parity check failed: max_diff={max_diff:.6e} (atol={config.atol}), "
            f"max_rel_diff={max_rel_diff:.6e} (rtol={config.rtol})"
        )
    
    if config.verbose:
        print(f"Paged output shape: {paged_output.shape}")
        print(f"Linear output shape: {linear_output.shape}")
        print(f"Max absolute diff: {max_diff:.6e}")
        print(f"Mean absolute diff: {mean_diff:.6e}")
        print(f"Max relative diff: {max_rel_diff:.6e}")
        print(f"Valid: {is_valid}")
    
    return ParityValidationResult(
        is_valid=is_valid,
        max_diff=max_diff,
        mean_diff=mean_diff,
        rel_diff=max_rel_diff,
        paged_output=paged_output,
        linear_output=linear_output,
        atol=config.atol,
        rtol=config.rtol,
        error_message=error_message,
    )


def validate_paged_v1_parity(
    query: NDArray[Any],
    k_cache: NDArray[Any],
    v_cache: NDArray[Any],
    block_tables: NDArray[Any],
    context_lens: NDArray[Any],
    scale: float | None = None,
    config: ValidationConfig | None = None,
) -> ParityValidationResult:
    """Validate paged_attention_v1 against linear attention.
    
    This is a convenience function that runs both paged and linear attention
    and validates their outputs match.
    
    Args:
        query: Query tensor [num_seqs, num_heads, head_dim]
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: Block indices [num_seqs, max_blocks_per_seq]
        context_lens: Context lengths [num_seqs]
        scale: Attention scale (default: 1/sqrt(head_dim))
        config: Validation configuration
        
    Returns:
        ParityValidationResult
    """
    from .attention import paged_attention_v1
    
    if config is None:
        config = ValidationConfig()
    
    num_seqs, num_heads, head_dim = query.shape
    block_size = k_cache.shape[1]
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Compute paged attention
    paged_output = paged_attention_v1(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
    )
    
    # Gather KV for linear attention
    k_linear, v_linear = _gather_kv_from_blocks(
        k_cache, v_cache, block_tables, context_lens, block_size
    )
    
    # Compute linear attention
    # Expand query for linear attention: [num_seqs, num_heads, 1, head_dim]
    q_expanded = np.expand_dims(query, axis=2)
    
    # Linear attention expects [num_seqs, num_heads, kv_len, head_dim]
    # k_linear is [num_seqs, max_context, num_kv_heads, head_dim]
    # Transpose to [num_seqs, num_kv_heads, max_context, head_dim]
    k_linear = np.transpose(k_linear, (0, 2, 1, 3))
    v_linear = np.transpose(v_linear, (0, 2, 1, 3))
    
    linear_output = compute_linear_attention(
        query=q_expanded,
        keys=k_linear,
        values=v_linear,
        scale=scale,
        is_causal=False,  # Decode doesn't need causal mask
    )
    
    # Squeeze output dimension
    linear_output = np.squeeze(linear_output, axis=2)
    
    return validate_paged_linear_parity(paged_output, linear_output, config)


def validate_paged_block_pool_parity(
    query: NDArray[Any],
    block_pool: NDArray[Any],
    block_tables: NDArray[Any],
    context_lens: NDArray[Any],
    scale: float,
    num_kv_heads: int,
    block_size: int = 16,
    config: ValidationConfig | None = None,
) -> ParityValidationResult:
    """Validate paged_attention (block pool) against linear attention.
    
    Args:
        query: Query tensor [num_seqs, num_heads, seq_len, head_dim]
        block_pool: Block pool [num_blocks, 2, block_size, num_kv_heads, head_dim]
        block_tables: Block indices [num_seqs, max_blocks_per_seq]
        context_lens: Context lengths [num_seqs]
        scale: Attention scale factor
        num_kv_heads: Number of KV heads
        block_size: Tokens per block
        config: Validation configuration
        
    Returns:
        ParityValidationResult
    """
    from .attention import paged_attention
    
    if config is None:
        config = ValidationConfig()
    
    # Compute paged attention
    paged_output = paged_attention(
        query=query,
        block_pool=block_pool,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )
    
    # Extract KV from block pool for linear attention
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size
    
    # Gather from block pool
    keys = []
    values = []
    for seq_idx in range(num_seqs):
        seq_keys = []
        seq_values = []
        ctx_len = int(context_lens[seq_idx])
        num_blocks_needed = (ctx_len + block_size - 1) // block_size
        
        for blk in range(num_blocks_needed):
            phys_block = int(block_tables[seq_idx, blk])
            seq_keys.append(block_pool[phys_block, 0])  # K is at index 0
            seq_values.append(block_pool[phys_block, 1])  # V is at index 1
        
        k_concat = np.concatenate(seq_keys, axis=0)[:ctx_len]  # [ctx_len, num_kv_heads, head_dim]
        v_concat = np.concatenate(seq_values, axis=0)[:ctx_len]
        
        # Pad to max_context
        k_padded = np.zeros((max_context, num_kv_heads, head_dim), dtype=block_pool.dtype)
        v_padded = np.zeros((max_context, num_kv_heads, head_dim), dtype=block_pool.dtype)
        k_padded[:ctx_len] = k_concat
        v_padded[:ctx_len] = v_concat
        
        keys.append(k_padded)
        values.append(v_padded)
    
    keys = np.stack(keys, axis=0)  # [num_seqs, max_context, num_kv_heads, head_dim]
    values = np.stack(values, axis=0)
    
    # Transpose to [num_seqs, num_kv_heads, max_context, head_dim]
    keys = np.transpose(keys, (0, 2, 1, 3))
    values = np.transpose(values, (0, 2, 1, 3))
    
    # Compute linear attention
    is_causal = seq_len > 1  # Prefill needs causal mask
    linear_output = compute_linear_attention(
        query=query,
        keys=keys,
        values=values,
        scale=scale,
        is_causal=is_causal,
    )
    
    return validate_paged_linear_parity(paged_output, linear_output, config)


class ParityValidator:
    """Stateful validator for running multiple parity checks.
    
    This class maintains configuration and statistics across multiple
    validation runs, useful for comprehensive test suites.
    
    Example:
        validator = ParityValidator(ValidationConfig(atol=1e-3))
        
        for config in test_configs:
            result = validator.validate_v1(
                query=config["query"],
                k_cache=config["k_cache"],
                v_cache=config["v_cache"],
                block_tables=config["block_tables"],
                context_lens=config["context_lens"],
            )
            
        stats = validator.get_statistics()
        print(f"Passed: {stats['passed']}/{stats['total']}")
    """
    
    def __init__(self, config: ValidationConfig | None = None):
        """Initialize validator with configuration.
        
        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()
        self.results: list[ParityValidationResult] = []
    
    def validate_v1(
        self,
        query: NDArray[Any],
        k_cache: NDArray[Any],
        v_cache: NDArray[Any],
        block_tables: NDArray[Any],
        context_lens: NDArray[Any],
        scale: float | None = None,
    ) -> ParityValidationResult:
        """Validate paged_attention_v1 and record result.
        
        Args:
            query: Query tensor
            k_cache: Key cache
            v_cache: Value cache
            block_tables: Block indices
            context_lens: Context lengths
            scale: Attention scale
            
        Returns:
            ParityValidationResult
        """
        result = validate_paged_v1_parity(
            query, k_cache, v_cache, block_tables, context_lens,
            scale, self.config
        )
        self.results.append(result)
        return result
    
    def validate_block_pool(
        self,
        query: NDArray[Any],
        block_pool: NDArray[Any],
        block_tables: NDArray[Any],
        context_lens: NDArray[Any],
        scale: float,
        num_kv_heads: int,
        block_size: int = 16,
    ) -> ParityValidationResult:
        """Validate paged_attention (block pool) and record result.
        
        Args:
            query: Query tensor
            block_pool: Block pool
            block_tables: Block indices
            context_lens: Context lengths
            scale: Attention scale
            num_kv_heads: Number of KV heads
            block_size: Tokens per block
            
        Returns:
            ParityValidationResult
        """
        result = validate_paged_block_pool_parity(
            query, block_pool, block_tables, context_lens,
            scale, num_kv_heads, block_size, self.config
        )
        self.results.append(result)
        return result
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics from all validation runs.
        
        Returns:
            Dictionary with statistics
        """
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}
        
        passed = sum(1 for r in self.results if r.is_valid)
        failed = len(self.results) - passed
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0.0,
            "max_diff": max(r.max_diff for r in self.results) if self.results else 0.0,
            "mean_diff": sum(r.mean_diff for r in self.results) / len(self.results) if self.results else 0.0,
        }
    
    def reset(self) -> None:
        """Clear all recorded results."""
        self.results.clear()
