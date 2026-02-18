"""Enhanced Paged Attention vs Linear Attention parity validation.

This module provides comprehensive validation tools with proper FP16/FP32
aware tolerances to ensure numerical equivalence between paged and linear
attention implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


@dataclass
class ParityConfig:
    """Configuration for parity validation with dtype-aware tolerances.
    
    Attributes:
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison  
        fp16_atol: Absolute tolerance for FP16 (default: 5e-3)
        fp16_rtol: Relative tolerance for FP16 (default: 5e-2)
        fp32_atol: Absolute tolerance for FP32 (default: 1e-5)
        fp32_rtol: Relative tolerance for FP32 (default: 1e-2)
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for Inf values
    """
    atol: float | None = None
    rtol: float | None = None
    fp16_atol: float = 5e-3
    fp16_rtol: float = 5e-2
    fp32_atol: float = 1e-5
    fp32_rtol: float = 1e-2
    check_nan: bool = True
    check_inf: bool = True
    
    def get_tolerances(self, dtype: np.dtype | str) -> tuple[float, float]:
        """Get appropriate tolerances for the given dtype.
        
        Args:
            dtype: NumPy dtype or string representation
            
        Returns:
            Tuple of (atol, rtol)
        """
        dtype_str = str(dtype)
        
        # Use explicit tolerances if provided
        if self.atol is not None and self.rtol is not None:
            return (self.atol, self.rtol)
        
        # Auto-select based on dtype
        if "float16" in dtype_str or "fp16" in dtype_str:
            return (self.fp16_atol, self.fp16_rtol)
        else:
            return (self.fp32_atol, self.fp32_rtol)


@dataclass 
class ParityResult:
    """Result of parity validation.
    
    Attributes:
        passed: Whether validation passed
        max_abs_diff: Maximum absolute difference
        mean_abs_diff: Mean absolute difference
        max_rel_diff: Maximum relative difference
        atol: Absolute tolerance used
        rtol: Relative tolerance used
        message: Human-readable result message
    """
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    atol: float
    rtol: float
    message: str
    
    def __str__(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"{status}: max_diff={self.max_abs_diff:.6e}, "
            f"rel_diff={self.max_rel_diff:.6e}, "
            f"atol={self.atol:.6e}, rtol={self.rtol:.6e}"
        )


def validate_parity(
    paged_output: NDArray[Any],
    linear_output: NDArray[Any],
    config: ParityConfig | None = None,
) -> ParityResult:
    """Validate parity between paged and linear attention outputs.
    
    Args:
        paged_output: Output from paged attention
        linear_output: Output from linear attention
        config: Validation configuration
        
    Returns:
        ParityResult with detailed comparison
    """
    if config is None:
        config = ParityConfig()
    
    # Determine dtype for tolerance selection
    dtype = paged_output.dtype
    atol, rtol = config.get_tolerances(dtype)
    
    # Check for NaN
    if config.check_nan:
        paged_nan = np.isnan(paged_output).any()
        linear_nan = np.isnan(linear_output).any()
        if paged_nan or linear_nan:
            return ParityResult(
                passed=False,
                max_abs_diff=float("nan"),
                mean_abs_diff=float("nan"),
                max_rel_diff=float("nan"),
                atol=atol,
                rtol=rtol,
                message=f"NaN detected: paged={paged_nan}, linear={linear_nan}",
            )
    
    # Check for Inf
    if config.check_inf:
        paged_inf = np.isinf(paged_output).any()
        linear_inf = np.isinf(linear_output).any()
        if paged_inf or linear_inf:
            return ParityResult(
                passed=False,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                atol=atol,
                rtol=rtol,
                message=f"Inf detected: paged={paged_inf}, linear={linear_inf}",
            )
    
    # Compute differences
    abs_diff = np.abs(paged_output - linear_output)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    
    # Compute relative difference
    linear_abs = np.abs(linear_output)
    rel_diff = np.where(
        linear_abs > 1e-10,
        abs_diff / linear_abs,
        abs_diff,
    )
    max_rel_diff = float(np.max(rel_diff))
    
    # Check tolerance
    passed = (max_abs_diff <= atol) or (max_rel_diff <= rtol)
    
    if passed:
        message = (
            f"Parity check passed: max_diff={max_abs_diff:.6e} <= {atol:.6e} or "
            f"rel_diff={max_rel_diff:.6e} <= {rtol:.6e}"
        )
    else:
        message = (
            f"Parity check failed: max_diff={max_abs_diff:.6e} > {atol:.6e} and "
            f"rel_diff={max_rel_diff:.6e} > {rtol:.6e}"
        )
    
    return ParityResult(
        passed=passed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        atol=atol,
        rtol=rtol,
        message=message,
    )


def run_paged_v1_parity_test(
    num_seqs: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    seq_len: int = 16,
    block_size: int = 16,
    dtype: np.dtype = np.float16,
    seed: int = 42,
) -> ParityResult:
    """Run a single paged attention v1 parity test.
    
    This is a convenience function for quick validation testing.
    
    Args:
        num_seqs: Number of sequences
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Head dimension
        seq_len: Sequence length
        block_size: Block size for paging
        dtype: Data type for tensors
        seed: Random seed
        
    Returns:
        ParityResult from the validation
    """
    from metal_marlin.paged.attention import paged_attention_v1
    from metal_marlin.paged.validation import compute_linear_attention
    
    np.random.seed(seed)
    
    # Create query [num_seqs, num_heads, head_dim]
    query = np.random.randn(num_seqs, num_heads, head_dim).astype(dtype)
    
    # Create KV tensors [num_seqs, seq_len, num_kv_heads, head_dim]
    k = np.random.randn(num_seqs, seq_len, num_kv_heads, head_dim).astype(dtype)
    v = np.random.randn(num_seqs, seq_len, num_kv_heads, head_dim).astype(dtype)
    
    # Create paged KV cache
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * blocks_per_seq
    
    k_cache = np.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    v_cache = np.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    
    block_tables = np.zeros((num_seqs, blocks_per_seq), dtype=np.int32)
    
    # Populate cache and block tables
    for seq_idx in range(num_seqs):
        for blk_idx in range(blocks_per_seq):
            phys_idx = seq_idx * blocks_per_seq + blk_idx
            block_tables[seq_idx, blk_idx] = phys_idx
        
        for token_idx in range(seq_len):
            local_block_idx = token_idx // block_size
            slot_idx = token_idx % block_size
            phys_block_idx = seq_idx * blocks_per_seq + local_block_idx
            
            k_cache[phys_block_idx, slot_idx] = k[seq_idx, token_idx]
            v_cache[phys_block_idx, slot_idx] = v[seq_idx, token_idx]
    
    context_lens = np.full((num_seqs,), seq_len, dtype=np.int32)
    scale = 1.0 / math.sqrt(head_dim)
    
    # Run paged attention
    paged_output = paged_attention_v1(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
    )
    
    # Run linear attention
    q_expanded = np.expand_dims(query, axis=2)  # [num_seqs, num_heads, 1, head_dim]
    
    # Gather KV for linear attention
    k_linear = np.zeros((num_seqs, seq_len, num_kv_heads, head_dim), dtype=dtype)
    v_linear = np.zeros((num_seqs, seq_len, num_kv_heads, head_dim), dtype=dtype)
    
    for seq_idx in range(num_seqs):
        for token_idx in range(seq_len):
            local_block_idx = token_idx // block_size
            slot_idx = token_idx % block_size
            phys_block_idx = seq_idx * blocks_per_seq + local_block_idx
            
            k_linear[seq_idx, token_idx] = k_cache[phys_block_idx, slot_idx]
            v_linear[seq_idx, token_idx] = v_cache[phys_block_idx, slot_idx]
    
    # Transpose for linear attention
    k_linear_t = np.transpose(k_linear, (0, 2, 1, 3))  # [num_seqs, num_kv_heads, seq_len, head_dim]
    v_linear_t = np.transpose(v_linear, (0, 2, 1, 3))
    
    linear_output = compute_linear_attention(
        query=q_expanded,
        keys=k_linear_t,
        values=v_linear_t,
        scale=scale,
        is_causal=False,
    )
    linear_output = np.squeeze(linear_output, axis=2)  # [num_seqs, num_heads, head_dim]
    
    # Validate
    config = ParityConfig()
    return validate_parity(paged_output, linear_output, config)


def run_comprehensive_parity_suite(
    configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a comprehensive suite of parity tests.
    
    Args:
        configs: List of test configurations. If None, uses default configs.
        
    Returns:
        Dictionary with test results and statistics
    """
    if configs is None:
        # Default test configurations
        configs = [
            # FP16 tests
            {"num_seqs": 1, "num_heads": 4, "num_kv_heads": 4, "dtype": np.float16, "name": "fp16_single_seq"},
            {"num_seqs": 2, "num_heads": 4, "num_kv_heads": 4, "dtype": np.float16, "name": "fp16_multi_seq"},
            {"num_seqs": 4, "num_heads": 8, "num_kv_heads": 8, "dtype": np.float16, "name": "fp16_batch4"},
            {"num_seqs": 1, "num_heads": 8, "num_kv_heads": 4, "dtype": np.float16, "name": "fp16_gqa"},
            # FP32 tests  
            {"num_seqs": 1, "num_heads": 4, "num_kv_heads": 4, "dtype": np.float32, "name": "fp32_single_seq"},
            {"num_seqs": 2, "num_heads": 4, "num_kv_heads": 4, "dtype": np.float32, "name": "fp32_multi_seq"},
        ]
    
    results = []
    passed = 0
    failed = 0
    
    for config in configs:
        name = config.pop("name", "unnamed")
        try:
            result = run_paged_v1_parity_test(**config)
            results.append({
                "name": name,
                "config": config,
                "result": result,
            })
            if result.passed:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results.append({
                "name": name,
                "config": config,
                "error": str(e),
            })
            failed += 1
    
    return {
        "total": len(configs),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(configs) if configs else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    # Run comprehensive suite when executed directly
    print("Running Paged Attention vs Linear Attention Parity Validation")
    print("=" * 70)
    
    results = run_comprehensive_parity_suite()
    
    print(f"\nResults: {results['passed']}/{results['total']} passed "
          f"({results['pass_rate']*100:.1f}%)")
    print()
    
    for r in results["results"]:
        name = r["name"]
        if "error" in r:
            print(f"{name}: ERROR - {r['error']}")
        else:
            result = r["result"]
            status = "✓" if result.passed else "✗"
            print(f"{status} {name}: {result}")
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if results["failed"] == 0 else 1)
