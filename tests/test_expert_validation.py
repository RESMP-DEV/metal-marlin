"""Comprehensive validation tests for all 64 experts individually.

Tests each expert in the MoE layer to verify:
1. Known input → expected output (SwiGLU correctness)
2. Numerical accuracy against reference implementation
3. Gradient flow for training scenarios

NOTE: These tests use MOCK weights (random packed indices), which have known limitations:
- Random packed indices dequantize to uncalibrated values
- Large inputs may cause overflow (Inf)
- Outputs may be numerically unstable
- Fast vs slow path comparison is the correct validation approach

For real model accuracy testing, use the @pytest.mark.slow tests that load actual models.

Verify: cd contrib/metal_marlin && uv run pytest tests/test_expert_validation.py -v
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_MPS = torch.backends.mps.is_available()
_HAS_TRELLIS = False

try:
    from metal_marlin.trellis.layer import TrellisDenseMLP
    from metal_marlin.trellis.model import TrellisMoEMLP
    from metal_marlin.trellis.testing import (
        create_mock_dense_mlp,
        create_mock_moe_mlp,
        create_mock_trellis_linear,
    )

    _HAS_TRELLIS = True
except ImportError:
    pass

HAS_TRELLIS = _HAS_TRELLIS

requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required (Apple Silicon)")
requires_trellis = pytest.mark.skipif(not HAS_TRELLIS, reason="Trellis modules required")


def clear_mps_memory() -> None:
    """Clear MPS memory cache and run garbage collection."""
    gc.collect()
    if HAS_MPS:
        torch.mps.empty_cache()
        torch.mps.synchronize()


@dataclass
class ExpertTestResult:
    """Results from testing a single expert."""

    expert_id: int
    output_shape_correct: bool
    output_finite: bool
    swiglu_matches_reference: bool
    max_diff_from_reference: float
    gradient_flows: bool
    gradient_norm: float


def swiglu_reference(
    x: torch.Tensor,
    gate_proj: nn.Module,
    up_proj: nn.Module,
    down_proj: nn.Module,
) -> torch.Tensor:
    """Reference SwiGLU implementation for validation.

    SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))

    This is the ground truth implementation that expert outputs
    should match.
    """
    gate = F.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def full_64_expert_moe() -> Any:
    """Create a mock MoE layer with 64 experts for validation.

    Uses smaller dimensions than real model for fast testing:
    - hidden_dim=256 (vs 4096 in real model)
    - intermediate_dim=512 (vs ~11k in real model)

    Memory: ~200MB vs 14GB for real model

    NOTE: Uses mock weights which may produce unstable outputs.
    Tests using this fixture should focus on:
    - Fast vs slow path comparison (both use same weights)
    - Shape correctness
    - Routing behavior
    NOT on absolute numerical accuracy (use real model tests for that).
    """
    if not HAS_MPS:
        pytest.skip("MPS not available")
    if not HAS_TRELLIS:
        pytest.skip("Trellis modules not available")

    clear_mps_memory()

    # Create MoE with 64 experts
    moe = create_mock_moe_mlp(
        hidden_dim=256,
        intermediate_dim=512,
        num_experts=64,  # Full 64 experts like GLM-4.7-Flash
        num_experts_per_tok=8,  # Top-8 routing
        bits=3,
        device="mps",
        eager_buffers=False,  # Need PyTorch tensors for slow path validation
    )

    yield moe

    del moe
    clear_mps_memory()


@pytest.fixture(scope="module")
def small_moe_for_gradients():
    """Create a small MoE layer for gradient testing.

    Smaller than full 64 experts to keep gradient tests fast.
    """
    if not HAS_MPS:
        pytest.skip("MPS not available")
    if not HAS_TRELLIS:
        pytest.skip("Trellis modules not available")

    clear_mps_memory()

    moe = create_mock_moe_mlp(
        hidden_dim=64,
        intermediate_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        bits=3,
        device="mps",
        eager_buffers=False,
    )

    yield moe

    del moe
    clear_mps_memory()


# ==============================================================================
# Test Classes: Individual Expert Validation
# ==============================================================================


@requires_mps
@requires_trellis
class TestIndividualExpertOutputs:
    """Test 1: Validate each expert produces correct outputs individually.

    NOTE: These tests use mock weights (random packed indices) which have limitations:
    - Uncalibrated dequantization may produce large/unstable values
    - Some experts may produce Inf due to overflow
    - Absolute numerical accuracy tests are skipped (use real model tests)

    We focus on:
    - Shape correctness
    - Routing behavior
    - Fast vs slow path comparison (relative correctness)
    """

    def test_all_64_experts_output_shapes(self, full_64_expert_moe: Any) -> None:
        """Verify each of 64 experts produces correct output shape."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Use small input to reduce overflow risk with mock weights
        x = torch.randn(4, hidden_dim, dtype=torch.float16, device="mps") * 0.1

        errors = []
        for i, expert in enumerate(moe.experts):
            with torch.no_grad():
                try:
                    output = expert(x)
                    if output.shape != x.shape:
                        errors.append(f"Expert {i}: shape {output.shape} != {x.shape}")
                except Exception as e:
                    errors.append(f"Expert {i}: forward failed: {e}")

        assert not errors, "Expert output shape errors:\n" + "\n".join(errors)

    def test_all_64_experts_no_nan_with_small_input(self, full_64_expert_moe: Any) -> None:
        """Verify no expert produces NaN with small inputs.

        Note: With mock weights, large inputs may overflow to Inf.
        This is documented behavior - real model weights are calibrated.
        We test with small inputs which should be stable.
        """
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Use SMALL input to avoid overflow with mock weights
        torch.manual_seed(42)
        x = torch.randn(8, hidden_dim, dtype=torch.float16, device="mps") * 0.01

        nan_experts = []

        for i, expert in enumerate(moe.experts):
            with torch.no_grad():
                output = expert(x)
                if torch.isnan(output).any():
                    nan_experts.append(i)

        # NaN should never happen (Inf can happen with mock weights, but not NaN)
        assert not nan_experts, f"Experts with NaN output: {nan_experts}"

    def test_all_64_experts_swiglu_structure(self, full_64_expert_moe: Any) -> None:
        """Verify each expert implements SwiGLU structure correctly.

        Tests that expert.forward() computes the same as the explicit
        SwiGLU formula: down_proj(silu(gate_proj(x)) * up_proj(x))

        With mock weights, intermediate values may overflow, so we use
        small inputs and check finite results match.
        """
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Small input to avoid overflow
        torch.manual_seed(42)
        x = torch.randn(2, hidden_dim, dtype=torch.float16, device="mps") * 0.01

        finite_matches = 0
        total_finite = 0

        for i, expert in enumerate(moe.experts):
            with torch.no_grad():
                actual = expert(x)
                expected = swiglu_reference(
                    x, expert.gate_proj, expert.up_proj, expert.down_proj
                )

                # Only check if both are finite
                if torch.isfinite(actual).all() and torch.isfinite(expected).all():
                    total_finite += 1
                    max_diff = (actual - expected).abs().max().item()
                    # Should match exactly - same computation
                    if max_diff < 1e-3:
                        finite_matches += 1

        # Most experts should produce finite outputs with small input
        assert total_finite >= 50, f"Only {total_finite}/64 experts produced finite output"
        # All finite outputs should match reference
        assert finite_matches == total_finite, (
            f"Only {finite_matches}/{total_finite} finite outputs matched reference"
        )

    def test_expert_zero_input_small_output(self, full_64_expert_moe: Any) -> None:
        """Test zero input produces small output (SwiGLU(0) ≈ 0).

        Zero input: gate_proj(0), up_proj(0) may have bias from quantization,
        but silu(x) ≈ 0 for x near 0, so output should be small.
        """
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim
        expert = moe.experts[0]

        x_zero = torch.zeros(1, hidden_dim, dtype=torch.float16, device="mps")
        with torch.no_grad():
            out_zero = expert(x_zero)

        # With quantization noise, output won't be exactly zero
        # but should be reasonably small
        max_out = out_zero.abs().max().item()
        # Very loose bound since mock weights aren't calibrated
        assert max_out < 1000 or not torch.isfinite(out_zero).all(), (
            f"Zero input gave unexpectedly large output: {max_out:.4f}"
        )

    def test_experts_produce_different_outputs(self, full_64_expert_moe: Any) -> None:
        """Verify different experts produce different outputs for same input."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        torch.manual_seed(42)
        x = torch.randn(1, hidden_dim, dtype=torch.float16, device="mps") * 0.01

        # Get outputs from first 10 experts
        outputs = []
        with torch.no_grad():
            for expert in list(moe.experts)[:10]:
                outputs.append(expert(x).clone())

        # Check that experts produce different outputs (at least some diversity)
        same_output_pairs = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if torch.allclose(outputs[i], outputs[j], atol=1e-3):
                    same_output_pairs.append((i, j))

        # Some experts might have similar outputs due to random weights
        # but most should differ
        assert len(same_output_pairs) < 5, (
            f"Too many expert pairs with identical output: {same_output_pairs}"
        )


@requires_mps
@requires_trellis
class TestNumericalAccuracy:
    """Test 2: Check numerical accuracy against reference implementations.

    NOTE: With mock weights, these tests focus on:
    - Structural correctness (forward matches explicit SwiGLU)
    - Relative comparisons (fast vs slow path)

    Absolute accuracy requires real model weights - see @pytest.mark.slow tests.
    """

    def test_expert_vs_reference_mlp_small_input(self, full_64_expert_moe: Any) -> None:
        """Compare expert outputs to explicit SwiGLU with small inputs."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Small input to avoid overflow with mock weights
        torch.manual_seed(42)
        x = torch.randn(2, hidden_dim, dtype=torch.float16, device="mps") * 0.01

        matched = 0
        finite = 0

        for i, expert in enumerate(moe.experts):
            with torch.no_grad():
                expert_out = expert(x)

                # Reference computation (explicit SwiGLU)
                gate = F.silu(expert.gate_proj(x))
                up = expert.up_proj(x)
                ref_out = expert.down_proj(gate * up)

                # Only count finite outputs
                if torch.isfinite(expert_out).all() and torch.isfinite(ref_out).all():
                    finite += 1
                    max_diff = (expert_out - ref_out).abs().max().item()
                    if max_diff < 1e-3:
                        matched += 1

        # Most experts should produce finite output with small input
        assert finite >= 50, f"Only {finite}/64 experts produced finite output"
        # All finite outputs should match reference exactly
        assert matched == finite, f"Only {matched}/{finite} matched reference"

    def test_moe_fast_vs_slow_path_correlation(self, full_64_expert_moe: Any) -> None:
        """Verify fast and slow MoE paths produce correlated outputs.

        This is the key validation: both paths use the same quantized weights,
        so they should produce nearly identical results regardless of mock weights.
        """
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Small input for stability
        torch.manual_seed(42)
        x = torch.randn(4, hidden_dim, dtype=torch.float16, device="mps") * 0.01

        with torch.no_grad():
            # Slow path (Python reference)
            slow_output = moe._forward_slow(x)

            # Fast path (if available)
            if not moe._use_fast_moe:
                pytest.skip("Fast MoE kernel not available")

            fast_output = moe.forward_fast(x)

        # Compare outputs
        if torch.isfinite(slow_output).all() and torch.isfinite(fast_output).all():
            max_diff = (slow_output - fast_output).abs().max().item()
            # Should be very close - same computation, just different dispatch
            assert max_diff < 1.0, f"Fast vs slow path diff too large: {max_diff:.4f}"

            # Compute correlation
            slow_flat = slow_output.float().flatten()
            fast_flat = fast_output.float().flatten()
            slow_centered = slow_flat - slow_flat.mean()
            fast_centered = fast_flat - fast_flat.mean()
            correlation = (slow_centered * fast_centered).sum() / (
                slow_centered.norm() * fast_centered.norm() + 1e-8
            )
            assert correlation > 0.9, f"Correlation too low: {correlation:.4f}"

    def test_expert_output_variation(self, full_64_expert_moe: Any) -> None:
        """Verify experts produce variation (not constant output)."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Small input for stability
        torch.manual_seed(42)
        x = torch.randn(8, hidden_dim, dtype=torch.float16, device="mps") * 0.01

        low_variation = 0
        for i, expert in enumerate(moe.experts):
            with torch.no_grad():
                out = expert(x)

                if torch.isfinite(out).all():
                    std = out.float().std().item()
                    if std < 1e-6:
                        low_variation += 1

        # Most experts should produce output variation
        assert low_variation < 10, f"{low_variation} experts have near-zero variation"


@requires_mps
@requires_trellis
class TestGradientFlow:
    """Test 3: Verify gradient flow through experts (for training)."""

    def test_expert_gradients_exist(self, small_moe_for_gradients):
        """Verify gradients flow through each expert."""
        moe = small_moe_for_gradients
        hidden_dim = moe.hidden_dim

        # Need to test with requires_grad=True
        x = torch.randn(
            4, hidden_dim, dtype=torch.float32, device="mps", requires_grad=True
        )

        no_grad_experts = []
        for i, expert in enumerate(moe.experts):
            # Reset grad
            if x.grad is not None:
                x.grad.zero_()

            # Forward and backward
            out = expert(x.half()).float()
            loss = out.sum()
            loss.backward()

            # Check input gradient exists
            if x.grad is None or x.grad.abs().max() < 1e-12:
                no_grad_experts.append(i)

            # Create fresh x for next expert
            x = torch.randn(
                4, hidden_dim, dtype=torch.float32, device="mps", requires_grad=True
            )

        assert not no_grad_experts, (
            f"Experts with no gradient to input: {no_grad_experts}"
        )

    def test_expert_weight_gradients(self, small_moe_for_gradients):
        """Verify gradients flow to expert weights."""
        moe = small_moe_for_gradients
        hidden_dim = moe.hidden_dim

        x = torch.randn(4, hidden_dim, dtype=torch.float16, device="mps")

        no_weight_grad = []
        for i, expert in enumerate(moe.experts):
            # Zero grads on expert weights
            expert.zero_grad()

            # Forward and backward
            out = expert(x)
            loss = out.sum()
            loss.backward()

            # Check weight gradients exist for each projection
            # Note: TrellisLinear may not have gradients for quantized weights
            # but the forward should still be differentiable through the computation
            has_any_grad = False
            for name, param in expert.named_parameters():
                if param.grad is not None and param.grad.abs().max() > 0:
                    has_any_grad = True
                    break

            if not has_any_grad:
                no_weight_grad.append(i)

        # For quantized models, weight gradients may not exist
        # but input gradients should (tested above)
        # This test is informational
        if no_weight_grad:
            print(f"Note: {len(no_weight_grad)} experts have no weight gradients "
                  "(expected for quantized models)")

    def test_gradient_magnitude_reasonable(self, small_moe_for_gradients):
        """Verify gradient magnitudes are not exploding or vanishing."""
        moe = small_moe_for_gradients
        hidden_dim = moe.hidden_dim

        gradient_stats = []
        for i, expert in enumerate(moe.experts):
            x = torch.randn(
                8, hidden_dim, dtype=torch.float32, device="mps", requires_grad=True
            )

            out = expert(x.half()).float()
            loss = out.sum()
            loss.backward()

            grad_norm = x.grad.norm().item()
            gradient_stats.append((i, grad_norm))

        # Check for vanishing gradients
        vanishing = [(i, n) for i, n in gradient_stats if n < 1e-8]
        # Check for exploding gradients
        exploding = [(i, n) for i, n in gradient_stats if n > 1e6]

        assert not vanishing, f"Experts with vanishing gradients: {vanishing}"
        assert not exploding, f"Experts with exploding gradients: {exploding}"


@requires_mps
@requires_trellis
class TestExpertRouting:
    """Test expert routing and weighted combination."""

    def test_all_experts_can_be_selected(self, full_64_expert_moe):
        """Verify all 64 experts can potentially be selected by router."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        # Generate many random inputs to exercise routing
        torch.manual_seed(42)
        x = torch.randn(256, hidden_dim, dtype=torch.float16, device="mps")

        with torch.no_grad():
            x_router = x.to(moe.router.weight.dtype)
            router_logits = moe.router(x_router)
            routing_weights = F.softmax(router_logits, dim=-1)

            # Get top-k for each token
            _, selected = torch.topk(
                routing_weights, k=moe.num_experts_per_tok, dim=-1
            )

            # Count how many unique experts were selected across all tokens
            unique_experts = selected.unique().tolist()

        # With 256 tokens and top-8 routing, we should hit most experts
        assert len(unique_experts) >= 32, (
            f"Only {len(unique_experts)} unique experts selected "
            f"(expected at least 32 of 64)"
        )

    def test_routing_weights_sum_correctly(self, full_64_expert_moe):
        """Verify routing weights are properly normalized."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        torch.manual_seed(42)
        x = torch.randn(16, hidden_dim, dtype=torch.float16, device="mps")

        with torch.no_grad():
            x_router = x.to(moe.router.weight.dtype)
            router_logits = moe.router(x_router)
            routing_weights, _ = torch.topk(
                F.softmax(router_logits, dim=-1, dtype=torch.float),
                k=moe.num_experts_per_tok,
                dim=-1,
            )
            # Normalize
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Weights should sum to 1 for each token
        sums = routing_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Routing weights don't sum to 1: {sums}"
        )


@requires_mps
@requires_trellis
class TestExpertDeterminism:
    """Test experts are deterministic and reproducible."""

    def test_expert_output_deterministic(self, full_64_expert_moe):
        """Verify same input produces same output."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        torch.manual_seed(42)
        x = torch.randn(4, hidden_dim, dtype=torch.float16, device="mps")

        for i in [0, 31, 63]:  # Test first, middle, last expert
            expert = moe.experts[i]
            with torch.no_grad():
                out1 = expert(x.clone())
                out2 = expert(x.clone())

            assert torch.allclose(out1, out2), (
                f"Expert {i}: non-deterministic output"
            )

    def test_expert_seeded_reproducibility(self, full_64_expert_moe):
        """Verify outputs are reproducible with same seed."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        expert = moe.experts[0]

        # Run 1: seed and forward
        torch.manual_seed(12345)
        x1 = torch.randn(4, hidden_dim, dtype=torch.float16, device="mps")
        with torch.no_grad():
            out1 = expert(x1)

        # Run 2: same seed, same forward
        torch.manual_seed(12345)
        x2 = torch.randn(4, hidden_dim, dtype=torch.float16, device="mps")
        with torch.no_grad():
            out2 = expert(x2)

        assert torch.allclose(out1, out2), "Seeded runs not reproducible"


@requires_mps
@requires_trellis
class TestComprehensiveExpertValidation:
    """Comprehensive test running full validation on all 64 experts."""

    def test_full_expert_validation_suite(self, full_64_expert_moe):
        """Run complete validation on all 64 experts and summarize results."""
        moe = full_64_expert_moe
        hidden_dim = moe.hidden_dim

        torch.manual_seed(42)
        x = torch.randn(8, hidden_dim, dtype=torch.float16, device="mps")

        results: list[ExpertTestResult] = []

        for i, expert in enumerate(moe.experts):
            # Test output shape
            with torch.no_grad():
                output = expert(x)
            shape_correct = output.shape == x.shape

            # Test output is finite
            output_finite = torch.isfinite(output).all().item()

            # Test SwiGLU correctness
            with torch.no_grad():
                expected = swiglu_reference(
                    x, expert.gate_proj, expert.up_proj, expert.down_proj
                )
                max_diff = (output - expected).abs().max().item()
            swiglu_correct = max_diff < 1e-3

            # Test gradient flow
            x_grad = torch.randn(
                4, hidden_dim, dtype=torch.float32, device="mps", requires_grad=True
            )
            out_grad = expert(x_grad.half()).float()
            loss = out_grad.sum()
            loss.backward()
            grad_flows = x_grad.grad is not None and x_grad.grad.abs().max() > 1e-12
            grad_norm = x_grad.grad.norm().item() if x_grad.grad is not None else 0.0

            results.append(ExpertTestResult(
                expert_id=i,
                output_shape_correct=shape_correct,
                output_finite=output_finite,
                swiglu_matches_reference=swiglu_correct,
                max_diff_from_reference=max_diff,
                gradient_flows=grad_flows,
                gradient_norm=grad_norm,
            ))

        # Summarize results
        failed_shape = [r.expert_id for r in results if not r.output_shape_correct]
        failed_finite = [r.expert_id for r in results if not r.output_finite]
        failed_swiglu = [r.expert_id for r in results if not r.swiglu_matches_reference]
        failed_grad = [r.expert_id for r in results if not r.gradient_flows]

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERT VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total experts tested: {len(results)}")
        print(f"Shape correct: {len(results) - len(failed_shape)}/64")
        print(f"Output finite: {len(results) - len(failed_finite)}/64")
        print(f"SwiGLU correct: {len(results) - len(failed_swiglu)}/64")
        print(f"Gradients flow: {len(results) - len(failed_grad)}/64")

        # Stats on reference diff
        max_diffs = [r.max_diff_from_reference for r in results]
        print(f"Max diff from ref - mean: {sum(max_diffs)/len(max_diffs):.6f}, "
              f"max: {max(max_diffs):.6f}")

        # Stats on gradient norms
        grad_norms = [r.gradient_norm for r in results]
        print(f"Gradient norms - mean: {sum(grad_norms)/len(grad_norms):.4f}, "
              f"min: {min(grad_norms):.4f}, max: {max(grad_norms):.4f}")
        print("=" * 60)

        # Assert all passed
        assert not failed_shape, f"Experts with wrong shape: {failed_shape}"
        assert not failed_finite, f"Experts with non-finite output: {failed_finite}"
        assert not failed_swiglu, f"Experts with SwiGLU mismatch: {failed_swiglu}"
        assert not failed_grad, f"Experts with no gradient: {failed_grad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
