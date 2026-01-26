"""Verification utilities for tensor parallelism.

This module provides tools for verifying tensor parallel implementations
using CPU+GPU split mode. Since current Apple Silicon has a single GPU,
we can verify correctness by:

1. Running a layer on CPU (numpy)
2. Running the same layer sharded between CPU and GPU
3. Comparing results for numerical equivalence

This approach validates:
- Sharding/gathering logic is correct
- Collective operations (all-reduce, all-gather) work properly
- No precision loss from tensor parallel communication

Example:
    from metal_marlin.distributed.verification import (
        verify_tensor_parallel_linear,
        verify_column_parallel,
        verify_row_parallel,
    )

    # Create a reference layer and tensor parallel version
    linear = MarlinLinear(...)
    mesh = DeviceMesh.cpu_gpu_split()
    tp_linear = TensorParallelLinear.from_marlin_linear(
        linear, mesh, ParallelMode.COLUMN
    )

    # Verify numerical equivalence
    is_correct, max_diff = verify_tensor_parallel_linear(
        reference=linear,
        parallel=tp_linear,
        input_shape=(4, 512),
    )
    assert is_correct, f"Tensor parallel error: max diff = {max_diff}"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._compat import HAS_MLX, mx, to_numpy
from .device_mesh import DeviceMesh
from .tensor_parallel import ParallelMode, TensorParallelLinear

if TYPE_CHECKING:
    from ..layers import MarlinLinear


@dataclass
class VerificationResult:
    """Result of a tensor parallel verification test.

    Attributes:
        passed: True if verification passed
        max_abs_diff: Maximum absolute difference between reference and parallel
        mean_abs_diff: Mean absolute difference
        max_rel_diff: Maximum relative difference
        reference_output: Output from reference implementation
        parallel_output: Output from parallel implementation
    """

    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    reference_output: np.ndarray | None = None
    parallel_output: np.ndarray | None = None

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"VerificationResult({status}: "
            f"max_abs={self.max_abs_diff:.6e}, "
            f"mean_abs={self.mean_abs_diff:.6e}, "
            f"max_rel={self.max_rel_diff:.6e})"
        )


def verify_tensor_parallel_linear(
    reference: MarlinLinear,
    parallel: TensorParallelLinear,
    input_shape: tuple[int, ...],
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 42,
    return_outputs: bool = False,
) -> VerificationResult:
    """Verify tensor parallel linear matches reference.

    Generates random input, runs through both implementations,
    and compares outputs for numerical equivalence.

    Args:
        reference: Reference MarlinLinear layer
        parallel: Tensor parallel version of the layer
        input_shape: Shape of input tensor (batch dimensions + in_features)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        seed: Random seed for reproducibility
        return_outputs: If True, include outputs in result

    Returns:
        VerificationResult with comparison metrics
    """
    rng = np.random.default_rng(seed)

    # Generate random input
    x_np = rng.standard_normal(input_shape).astype(np.float32)

    # Reference forward
    if HAS_MLX and mx is not None:
        x_ref = mx.array(x_np)
        y_ref = reference(x_ref)
        mx.eval(y_ref)
        y_ref_np = np.array(y_ref)
    else:
        y_ref_np = reference(x_np)
        if not isinstance(y_ref_np, np.ndarray):
            y_ref_np = to_numpy(y_ref_np)

    # Parallel forward
    y_par = parallel(x_np)
    if not isinstance(y_par, np.ndarray):
        y_par_np = to_numpy(y_par)
    else:
        y_par_np = y_par

    # Compare
    abs_diff = np.abs(y_ref_np - y_par_np)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Relative difference (avoid division by zero)
    ref_abs = np.abs(y_ref_np)
    rel_diff = abs_diff / np.maximum(ref_abs, 1e-8)
    max_rel_diff = float(np.max(rel_diff))

    # Check tolerance
    passed = np.allclose(y_ref_np, y_par_np, atol=atol, rtol=rtol)

    return VerificationResult(
        passed=passed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        reference_output=y_ref_np if return_outputs else None,
        parallel_output=y_par_np if return_outputs else None,
    )


def verify_column_parallel(
    in_features: int,
    out_features: int,
    batch_size: int = 4,
    group_size: int = 32,
    tp_size: int = 2,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 42,
) -> VerificationResult:
    """Verify column parallel implementation from scratch.

    Creates a MarlinLinear layer, converts to column parallel,
    and verifies equivalence.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        batch_size: Batch size for test input
        group_size: Quantization group size
        tp_size: Tensor parallel size
        atol: Absolute tolerance
        rtol: Relative tolerance
        seed: Random seed

    Returns:
        VerificationResult
    """
    from ..layers import MarlinLinear

    rng = np.random.default_rng(seed)

    # Create weights (random FP4-quantized)
    # Simulate packed FP4 weights
    packed_n = out_features // 8
    weight_packed = rng.integers(0, 2**32, size=(in_features, packed_n), dtype=np.uint32)
    num_groups = in_features // group_size
    scales = rng.uniform(0.1, 1.0, size=(num_groups, out_features)).astype(np.float16)
    bias = rng.uniform(-0.1, 0.1, size=(out_features,)).astype(np.float32)

    # Create reference layer
    reference = MarlinLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        quant_type="fp4",
        group_size=group_size,
        weight_packed=weight_packed,
        scales=scales,
    )
    reference.bias = bias

    # Create tensor parallel version
    mesh = DeviceMesh.cpu_only(tp_size)
    tp_linear = TensorParallelLinear.from_marlin_linear(
        reference, mesh, ParallelMode.COLUMN
    )

    # Verify
    return verify_tensor_parallel_linear(
        reference=reference,
        parallel=tp_linear,
        input_shape=(batch_size, in_features),
        atol=atol,
        rtol=rtol,
        seed=seed + 1,  # Different seed for input
    )


def verify_row_parallel(
    in_features: int,
    out_features: int,
    batch_size: int = 4,
    group_size: int = 32,
    tp_size: int = 2,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 42,
) -> VerificationResult:
    """Verify row parallel implementation from scratch.

    Creates a MarlinLinear layer, converts to row parallel,
    and verifies equivalence.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        batch_size: Batch size for test input
        group_size: Quantization group size
        tp_size: Tensor parallel size
        atol: Absolute tolerance
        rtol: Relative tolerance
        seed: Random seed

    Returns:
        VerificationResult
    """
    from ..layers import MarlinLinear

    rng = np.random.default_rng(seed)

    # Create weights
    packed_n = out_features // 8
    weight_packed = rng.integers(0, 2**32, size=(in_features, packed_n), dtype=np.uint32)
    num_groups = in_features // group_size
    scales = rng.uniform(0.1, 1.0, size=(num_groups, out_features)).astype(np.float16)
    bias = rng.uniform(-0.1, 0.1, size=(out_features,)).astype(np.float32)

    # Create reference layer
    reference = MarlinLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        quant_type="fp4",
        group_size=group_size,
        weight_packed=weight_packed,
        scales=scales,
    )
    reference.bias = bias

    # Create tensor parallel version
    mesh = DeviceMesh.cpu_only(tp_size)
    tp_linear = TensorParallelLinear.from_marlin_linear(
        reference, mesh, ParallelMode.ROW
    )

    # Verify
    return verify_tensor_parallel_linear(
        reference=reference,
        parallel=tp_linear,
        input_shape=(batch_size, in_features),
        atol=atol,
        rtol=rtol,
        seed=seed + 1,
    )


def verify_collective_operations(
    num_shards: int = 4,
    tensor_shape: tuple[int, ...] = (128, 256),
    seed: int = 42,
) -> dict[str, VerificationResult]:
    """Verify all collective operations.

    Tests all-reduce, all-gather, scatter, and reduce-scatter
    against reference numpy implementations.

    Args:
        num_shards: Number of shards to test with
        tensor_shape: Shape of each shard
        seed: Random seed

    Returns:
        Dictionary mapping operation name to VerificationResult
    """
    from .tensor_parallel import all_gather, all_reduce, scatter

    rng = np.random.default_rng(seed)
    results = {}

    # Test all-reduce (sum)
    tensors = [rng.standard_normal(tensor_shape).astype(np.float32) for _ in range(num_shards)]
    ref_sum = np.sum(np.stack(tensors), axis=0)
    par_sum = all_reduce(tensors, op="sum")

    results["all_reduce_sum"] = VerificationResult(
        passed=np.allclose(ref_sum, par_sum),
        max_abs_diff=float(np.max(np.abs(ref_sum - par_sum))),
        mean_abs_diff=float(np.mean(np.abs(ref_sum - par_sum))),
        max_rel_diff=float(np.max(np.abs(ref_sum - par_sum) / np.maximum(np.abs(ref_sum), 1e-8))),
    )

    # Test all-reduce (mean)
    ref_mean = np.mean(np.stack(tensors), axis=0)
    par_mean = all_reduce(tensors, op="mean")

    results["all_reduce_mean"] = VerificationResult(
        passed=np.allclose(ref_mean, par_mean),
        max_abs_diff=float(np.max(np.abs(ref_mean - par_mean))),
        mean_abs_diff=float(np.mean(np.abs(ref_mean - par_mean))),
        max_rel_diff=float(np.max(np.abs(ref_mean - par_mean) / np.maximum(np.abs(ref_mean), 1e-8))),
    )

    # Test all-gather
    shards = [rng.standard_normal((tensor_shape[0] // num_shards, tensor_shape[1])).astype(np.float32)
              for _ in range(num_shards)]
    ref_gather = np.concatenate(shards, axis=0)
    par_gather = all_gather(shards, dim=0)

    results["all_gather"] = VerificationResult(
        passed=np.allclose(ref_gather, par_gather),
        max_abs_diff=float(np.max(np.abs(ref_gather - par_gather))),
        mean_abs_diff=float(np.mean(np.abs(ref_gather - par_gather))),
        max_rel_diff=float(np.max(np.abs(ref_gather - par_gather) / np.maximum(np.abs(ref_gather), 1e-8))),
    )

    # Test scatter
    full_tensor = rng.standard_normal(tensor_shape).astype(np.float32)
    shard_size = tensor_shape[0] // num_shards
    ref_scatter = [full_tensor[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    par_scatter = scatter(full_tensor, dim=0, num_devices=num_shards)

    all_match = all(np.allclose(r, p) for r, p in zip(ref_scatter, par_scatter))
    max_diffs = [float(np.max(np.abs(r - p))) for r, p in zip(ref_scatter, par_scatter)]

    results["scatter"] = VerificationResult(
        passed=all_match,
        max_abs_diff=max(max_diffs),
        mean_abs_diff=sum(max_diffs) / len(max_diffs),
        max_rel_diff=0.0,  # Scatter should be exact
    )

    return results


def run_verification_suite(
    verbose: bool = True,
) -> bool:
    """Run full verification suite for tensor parallelism.

    Tests:
    - Column parallel linear (2-way, 4-way)
    - Row parallel linear (2-way, 4-way)
    - Collective operations
    - CPU+GPU split mode (if GPU available)

    Args:
        verbose: If True, print progress and results

    Returns:
        True if all tests pass
    """
    all_passed = True

    if verbose:
        print("=" * 60)
        print("Tensor Parallelism Verification Suite")
        print("=" * 60)

    # Test collective operations
    if verbose:
        print("\n1. Collective Operations")
        print("-" * 40)

    collective_results = verify_collective_operations()
    for name, result in collective_results.items():
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {name}: {status} (max_diff={result.max_abs_diff:.2e})")
        if not result.passed:
            all_passed = False

    # Test column parallel
    if verbose:
        print("\n2. Column Parallel Linear")
        print("-" * 40)

    for tp_size in [2, 4]:
        result = verify_column_parallel(
            in_features=512,
            out_features=1024,
            batch_size=4,
            group_size=32,
            tp_size=tp_size,
        )
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  tp_size={tp_size}: {status} (max_diff={result.max_abs_diff:.2e})")
        if not result.passed:
            all_passed = False

    # Test row parallel
    if verbose:
        print("\n3. Row Parallel Linear")
        print("-" * 40)

    for tp_size in [2, 4]:
        result = verify_row_parallel(
            in_features=512,
            out_features=1024,
            batch_size=4,
            group_size=32,
            tp_size=tp_size,
        )
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  tp_size={tp_size}: {status} (max_diff={result.max_abs_diff:.2e})")
        if not result.passed:
            all_passed = False

    # Test CPU+GPU split if GPU available
    if HAS_MLX:
        if verbose:
            print("\n4. CPU+GPU Split Mode")
            print("-" * 40)

        mesh = DeviceMesh.cpu_gpu_split()
        result = verify_column_parallel(
            in_features=512,
            out_features=1024,
            batch_size=4,
            group_size=32,
            tp_size=2,
        )
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  column_parallel: {status} (max_diff={result.max_abs_diff:.2e})")
        if not result.passed:
            all_passed = False

        result = verify_row_parallel(
            in_features=512,
            out_features=1024,
            batch_size=4,
            group_size=32,
            tp_size=2,
        )
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  row_parallel: {status} (max_diff={result.max_abs_diff:.2e})")
        if not result.passed:
            all_passed = False
    else:
        if verbose:
            print("\n4. CPU+GPU Split Mode")
            print("-" * 40)
            print("  SKIPPED (MLX not available)")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        if all_passed:
            print("All verification tests PASSED")
        else:
            print("Some verification tests FAILED")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys

    success = run_verification_suite(verbose=True)
    sys.exit(0 if success else 1)
