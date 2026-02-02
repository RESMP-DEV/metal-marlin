"""Test RMSNorm FP16 overflow handling.

Verifies that RMSNorm correctly handles large input values that would overflow
in FP16 precision (values like [-1000, 1200] from q_a_proj output).

The fix: variance computation uses FP32 accumulation via x.float().pow(2).mean()
to prevent overflow when squared values exceed FP16 max (~65504).
"""

import pytest
import torch

from metal_marlin.transformer import RMSNorm


@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TestRMSNormOverflow:
    """Test suite for RMSNorm FP16 overflow handling."""

    def test_rmsnorm_large_values_non_zero_output(self, device):
        """RMSNorm should produce non-zero output for large FP16 input values.

        Input range [-1000, 1200] is typical for q_a_proj outputs in MLA attention.
        These values cause x**2 to overflow in FP16 (1000**2 = 1M > 65504).

        Expected behavior: Non-zero normalized output due to FP32 accumulation.
        Bug behavior (if unfixed): All zeros due to rsqrt(inf) = 0.
        """
        hidden_size = 768
        eps = 1e-6

        norm = RMSNorm(hidden_size, eps=eps, device=device)

        # Input with values in range [-1000, 1200] (typical q_a_proj output)
        batch_size = 4
        seq_len = 16
        x = torch.empty(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
        x.uniform_(-1000, 1200)

        with torch.no_grad():
            output = norm(x)

        # Output should be non-zero
        assert not torch.all(output == 0), (
            f"RMSNorm produced all zeros for input range [{x.min():.1f}, {x.max():.1f}]. "
            "FP16 overflow may not be handled correctly."
        )

        # Output should be finite
        assert torch.isfinite(output).all(), "RMSNorm output contains inf/nan"

        # After normalization, values should be in a reasonable range (O(1) magnitude)
        assert output.abs().mean() < 10, (
            f"RMSNorm output has unexpectedly large magnitude: mean abs = {output.abs().mean():.2f}"
        )

    def test_rmsnorm_value_1000_works(self, device):
        """Test RMSNorm with uniform large value (1000).

        1000**2 = 1,000,000 which far exceeds FP16 max (65504).
        With FP32 accumulation, this should still produce valid output.
        """
        hidden_size = 768
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps=eps, device=device)

        x = torch.full((1, 1, hidden_size), 1000.0, device=device, dtype=torch.float16)

        with torch.no_grad():
            output = norm(x)

        # Output should be non-zero and finite
        assert not torch.all(output == 0), "RMSNorm produced zeros for value=1000"
        assert torch.isfinite(output).all(), "RMSNorm output contains inf/nan"

        # For uniform input, RMSNorm should normalize to ~1 (times weight)
        # Since weight is initialized to ones, output should be ~1
        expected_magnitude = 1.0
        actual_magnitude = output.abs().mean().item()
        assert abs(actual_magnitude - expected_magnitude) < 0.1, (
            f"Expected output magnitude ~{expected_magnitude}, got {actual_magnitude:.4f}"
        )

    def test_rmsnorm_fp32_accumulation_verified(self, device):
        """Verify that FP32 accumulation prevents overflow.

        Manually compute what would happen with naive FP16 vs FP32 accumulation.
        """
        hidden_size = 768
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps=eps, device=device)

        x = torch.full((1, 1, hidden_size), 500.0, device=device, dtype=torch.float16)

        # Naive FP16 computation would overflow
        x_squared_fp16 = x**2  # 500**2 = 250000 > 65504 -> inf in FP16
        naive_overflow = torch.isinf(x_squared_fp16).any()

        # But FP32 accumulation prevents this
        x_squared_fp32 = x.float().pow(2)  # No overflow in FP32
        fp32_finite = torch.isfinite(x_squared_fp32).all()

        assert naive_overflow, "Test setup error: FP16 squaring should overflow"
        assert fp32_finite, "FP32 squaring should not overflow"

        # RMSNorm should produce valid output
        with torch.no_grad():
            output = norm(x)

        assert not torch.all(output == 0), "RMSNorm should handle FP16 overflow"
        assert torch.isfinite(output).all(), "RMSNorm output should be finite"

    def test_rmsnorm_fp32_input_baseline(self, device):
        """Verify RMSNorm works correctly with FP32 input as baseline."""
        hidden_size = 768
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps=eps, device=device)

        x = torch.empty(4, 16, hidden_size, device=device, dtype=torch.float32)
        x.uniform_(-1000, 1200)

        with torch.no_grad():
            output = norm(x)

        assert not torch.all(output == 0), "FP32 input should produce non-zero output"
        assert torch.isfinite(output).all(), "FP32 output should be finite"
        assert output.dtype == torch.float32, "FP32 dtype should be preserved"

    def test_rmsnorm_small_values_fp16(self, device):
        """Verify RMSNorm works correctly with small FP16 values (no overflow concern)."""
        hidden_size = 768
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps=eps, device=device)

        x = torch.randn(4, 16, hidden_size, device=device, dtype=torch.float16)

        with torch.no_grad():
            output = norm(x)

        assert not torch.all(output == 0), "Small value output should be non-zero"
        assert torch.isfinite(output).all(), "Small value output should be finite"

    def test_rmsnorm_mixed_magnitudes(self, device):
        """Test RMSNorm with mixed small and large values."""
        hidden_size = 768
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps=eps, device=device)

        x = torch.zeros(1, 4, hidden_size, device=device, dtype=torch.float16)
        x[0, 0, :] = 0.1     # Small values
        x[0, 1, :] = 10.0    # Medium values
        x[0, 2, :] = 500.0   # Large values (would overflow squared in FP16)
        x[0, 3, :] = 1000.0  # Very large values

        with torch.no_grad():
            output = norm(x)

        # Each position should have non-zero output
        for pos in range(4):
            assert not torch.all(output[0, pos, :] == 0), (
                f"Position {pos} with input {x[0, pos, 0].item()} should not be zero"
            )

        assert torch.isfinite(output).all(), "All outputs should be finite"

    def test_rmsnorm_output_dtype(self, device):
        """RMSNorm output dtype depends on weight dtype.

        The weight is initialized as FP32 by default, so output will be FP32
        even with FP16 input due to weight * x promotion.
        """
        hidden_size = 768
        norm = RMSNorm(hidden_size=hidden_size, device=device)

        x = torch.randn(1, 8, hidden_size, device=device, dtype=torch.float16)
        output = norm(x)

        # Output is FP32 because weight is FP32 (default torch.ones dtype)
        # This is expected behavior: weight * x promotes to FP32
        assert output.dtype == torch.float32, (
            f"Expected FP32 output (weight is FP32), got {output.dtype}"
        )
        assert torch.isfinite(output).all(), "Output should be finite"

    def test_rmsnorm_boundary_values(self, device):
        """Test RMSNorm at the FP16 overflow boundary.

        FP16 max is 65504. sqrt(65504) ≈ 256.
        Values > 256 have x**2 > 65504 which overflows in FP16.
        Both should produce valid output with FP32 accumulation.
        """
        hidden_size = 768
        eps = 1e-6
        norm = RMSNorm(hidden_size, eps=eps, device=device)

        # Below boundary: 250**2 = 62500 < 65504
        x_below = torch.full((1, 1, hidden_size), 250.0, device=device, dtype=torch.float16)

        # Above boundary: 260**2 = 67600 > 65504
        x_above = torch.full((1, 1, hidden_size), 260.0, device=device, dtype=torch.float16)

        with torch.no_grad():
            output_below = norm(x_below)
            output_above = norm(x_above)

        # Both should produce valid non-zero output
        assert not torch.all(output_below == 0), "Value 250 should work"
        assert not torch.all(output_above == 0), "Value 260 should work (with FP32 accum)"

        assert torch.isfinite(output_below).all(), "Value 250 output should be finite"
        assert torch.isfinite(output_above).all(), "Value 260 output should be finite"

        # Both should normalize to approximately the same magnitude
        # since RMSNorm normalizes by the RMS value
        mag_below = output_below.abs().mean().item()
        mag_above = output_above.abs().mean().item()
        assert abs(mag_below - mag_above) < 0.1, (
            f"Normalized magnitudes should be similar: {mag_below:.4f} vs {mag_above:.4f}"
        )
