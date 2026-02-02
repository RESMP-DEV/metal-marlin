"""Regression tests for TrellisLinear fused GEMM accuracy.

Verifies that the fused dequantization+GEMM kernels produce outputs numerically
close to the explicit dequant + matmul reference implementation.

This is a regression test to ensure changes to the Metal kernels don't
introduce accuracy regressions.
"""

from __future__ import annotations

import pytest
import torch

HAS_MPS = torch.backends.mps.is_available()

try:
    from metal_marlin.quantization.trellis_codebook import TrellisCodebook
    from metal_marlin.trellis.linear import TrellisLinear

    _HAS_TRELLIS = True
except ImportError:
    _HAS_TRELLIS = False
    TrellisLinear = None  # type: ignore[assignment,misc]
    TrellisCodebook = None  # type: ignore[assignment,misc]


requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required")
requires_trellis = pytest.mark.skipif(not _HAS_TRELLIS, reason="Trellis modules required")


def create_test_trellis_linear(
    in_features: int,
    out_features: int,
    bits: int = 3,
    device: str = "mps",
    random_weights: bool = False,
) -> TrellisLinear:
    """Create a TrellisLinear layer for testing with correct buffer shapes.

    Uses the direct __init__ constructor which creates properly-shaped buffers,
    then optionally randomizes the packed data.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Quantization bits (2, 3, or 4).
        device: Target device.
        random_weights: If True, use random packed indices. If False, use zeros.

    Returns:
        TrellisLinear ready for testing.
    """
    assert TrellisLinear is not None
    assert TrellisCodebook is not None

    layer = TrellisLinear(in_features, out_features, bits, device=device)

    # Get the pre-computed codebook grid
    codebook = TrellisCodebook(bits=bits)
    grid = torch.from_numpy(codebook.get_grid()).float()
    layer.grid = grid.to(device)

    if random_weights:
        # Randomize packed indices for more realistic testing
        layer.packed_indices = torch.randint(
            0,
            256,
            layer.packed_indices.shape,
            dtype=torch.uint8,
            device=device,
        )
        # Randomize scales (small positive values)
        n_groups = layer.scales.shape[0]
        layer.scales = (
            torch.rand(n_groups, out_features, dtype=torch.float32, device=device) * 0.1 + 0.01
        )
        # Randomize sign vectors
        layer.su = torch.where(
            torch.rand(in_features, device=device) > 0.5,
            torch.ones(in_features, device=device),
            -torch.ones(in_features, device=device),
        )
        layer.sv = torch.where(
            torch.rand(out_features, device=device) > 0.5,
            torch.ones(out_features, device=device),
            -torch.ones(out_features, device=device),
        )
    else:
        # Zero packed indices for deterministic baseline
        layer.packed_indices.zero_()
        # Unit scales for predictable output
        layer.scales.fill_(1.0)
        # Unit sign vectors
        layer.su.fill_(1.0)
        layer.sv.fill_(1.0)

    return layer


@requires_mps
@requires_trellis
class TestTrellisLinearAccuracy:
    """Tests for TrellisLinear fused GEMM vs explicit dequant accuracy."""

    def test_trellis_linear_matches_explicit_dequant(self):
        """Verify fused GEMM matches explicit dequant + matmul."""
        torch.manual_seed(42)
        linear = create_test_trellis_linear(256, 128, bits=3, device="mps")
        x = torch.randn(1, 256, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)
            explicit = x @ linear.dequantize().T.to(x.dtype)

        # Should match within fp16 precision
        torch.testing.assert_close(fused, explicit, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "in_features,out_features",
        [
            (64, 32),
            (256, 128),
            (2048, 1536),
        ],
        ids=["64x32", "256x128", "2048x1536"],
    )
    def test_accuracy_different_sizes(self, in_features: int, out_features: int):
        """Test accuracy across different weight matrix sizes."""
        torch.manual_seed(42)
        linear = create_test_trellis_linear(in_features, out_features, bits=3, device="mps")
        x = torch.randn(1, in_features, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)
            explicit = x @ linear.dequantize().T.to(x.dtype)

        torch.testing.assert_close(
            fused,
            explicit,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Mismatch for size ({in_features}, {out_features})",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16], ids=["batch1", "batch4", "batch16"])
    def test_accuracy_different_batch_sizes(self, batch_size: int):
        """Test accuracy across different batch sizes."""
        torch.manual_seed(42)
        in_features, out_features = 256, 128
        linear = create_test_trellis_linear(in_features, out_features, bits=3, device="mps")
        x = torch.randn(batch_size, in_features, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)
            explicit = x @ linear.dequantize().T.to(x.dtype)

        torch.testing.assert_close(
            fused,
            explicit,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Mismatch for batch size {batch_size}",
        )

    @pytest.mark.parametrize("bits", [2, 3, 4], ids=["2bit", "3bit", "4bit"])
    def test_accuracy_different_bit_widths(self, bits: int):
        """Test accuracy across different quantization bit widths."""
        torch.manual_seed(42)
        in_features, out_features = 256, 128
        linear = create_test_trellis_linear(in_features, out_features, bits=bits, device="mps")
        x = torch.randn(1, in_features, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)
            explicit = x @ linear.dequantize().T.to(x.dtype)

        torch.testing.assert_close(
            fused,
            explicit,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Mismatch for {bits}-bit quantization",
        )

    @pytest.mark.parametrize(
        "in_features,out_features,batch_size,bits",
        [
            # 2-bit
            (64, 32, 1, 2),
            (64, 32, 4, 2),
            (64, 32, 16, 2),
            (256, 128, 1, 2),
            (256, 128, 4, 2),
            (256, 128, 16, 2),
            (2048, 1536, 1, 2),
            (2048, 1536, 4, 2),
            (2048, 1536, 16, 2),
            # 3-bit
            (64, 32, 1, 3),
            (64, 32, 4, 3),
            (64, 32, 16, 3),
            (256, 128, 1, 3),
            (256, 128, 4, 3),
            (256, 128, 16, 3),
            (2048, 1536, 1, 3),
            (2048, 1536, 4, 3),
            (2048, 1536, 16, 3),
            # 4-bit
            (64, 32, 1, 4),
            (64, 32, 4, 4),
            (64, 32, 16, 4),
            (256, 128, 1, 4),
            (256, 128, 4, 4),
            (256, 128, 16, 4),
            (2048, 1536, 1, 4),
            (2048, 1536, 4, 4),
            (2048, 1536, 16, 4),
        ],
        ids=[
            f"{i}x{o}_b{b}_{bits}bit"
            for bits in [2, 3, 4]
            for i, o in [(64, 32), (256, 128), (2048, 1536)]
            for b in [1, 4, 16]
        ],
    )
    def test_accuracy_full_matrix(
        self, in_features: int, out_features: int, batch_size: int, bits: int
    ):
        """Full parametric test: all size/batch/bit combinations."""
        torch.manual_seed(42)
        linear = create_test_trellis_linear(in_features, out_features, bits=bits, device="mps")
        x = torch.randn(batch_size, in_features, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)
            explicit = x @ linear.dequantize().T.to(x.dtype)

        torch.testing.assert_close(
            fused,
            explicit,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Mismatch: ({in_features}, {out_features}), batch={batch_size}, bits={bits}",
        )

    def test_determinism(self):
        """Verify fused output is deterministic across runs."""
        torch.manual_seed(42)
        linear = create_test_trellis_linear(256, 128, bits=3, device="mps")
        x = torch.randn(4, 256, dtype=torch.float16, device="mps")

        with torch.no_grad():
            out1 = linear(x)
            out2 = linear(x)

        torch.testing.assert_close(out1, out2, rtol=0, atol=0)

    def test_output_finite(self):
        """Verify outputs are finite (no NaN/Inf)."""
        torch.manual_seed(42)
        linear = create_test_trellis_linear(256, 128, bits=3, device="mps")
        x = torch.randn(4, 256, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)

        assert torch.isfinite(fused).all(), "Fused output contains NaN or Inf"

    def test_correlation_high(self):
        """Verify fused and explicit outputs are highly correlated."""
        torch.manual_seed(42)
        linear = create_test_trellis_linear(256, 128, bits=3, device="mps")
        x = torch.randn(8, 256, dtype=torch.float16, device="mps")

        with torch.no_grad():
            fused = linear(x)
            explicit = x @ linear.dequantize().T.to(x.dtype)

        # Pearson correlation
        fused_flat = fused.float().flatten()
        explicit_flat = explicit.float().flatten()
        fused_centered = fused_flat - fused_flat.mean()
        explicit_centered = explicit_flat - explicit_flat.mean()
        correlation = (fused_centered * explicit_centered).sum() / (
            fused_centered.norm() * explicit_centered.norm() + 1e-8
        )

        assert correlation > 0.99, f"Correlation too low: {correlation:.4f}"
