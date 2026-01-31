"""Minimal reproduction case for TrellisLinear NaN bug.

The bug manifests on expansion layers (in_features < out_features)
like q_b_proj (768 -> 5120) in GLM-4.7-Flash-Trellis-3bpw.

Reference: dispatch_gemm_trellis_decode in metal_marlin/trellis/dispatch.py
"""

from pathlib import Path

import pytest
import torch

from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS

MODEL_PATH = Path(__file__).parents[1] / "models" / "GLM-4.7-Flash-Trellis-3bpw"
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")


def require_metal():
    """Skip test if Metal/MPS is not available."""
    if not (HAS_MPS and HAS_METAL):
        pytest.skip("Metal/MPS not available")


class TestTrellisNaN:
    """Minimal reproduction for NaN bug in expansion layers."""

    def test_q_b_proj_nan(self):
        """Test q_b_proj expansion layer (768 -> 5120) produces NaN.

        This is the minimal reproduction case for the NaN bug.
        The kernel produces NaN for this specific expansion layer shape.
        """
        require_metal()

        from metal_marlin.trellis.linear import TrellisLinear
        from metal_marlin.trellis.loader import TrellisModelLoader

        # Load only the q_b_proj weight from layer 0
        loader = TrellisModelLoader(str(MODEL_PATH))
        weight = loader.load_weight(0, "self_attn.q_b_proj")

        # Verify shape: K=5120 (out_features), N=768 (in_features)
        # This is an expansion layer
        K, N = weight.original_shape
        assert K == 5120, f"Expected out_features=5120, got {K}"
        assert N == 768, f"Expected in_features=768, got {N}"
        assert weight.bits == 3, f"Expected bits=3, got {weight.bits}"

        # Create TrellisLinear from the loaded weight
        linear = TrellisLinear.from_trellis_weight(weight, device="mps")

        # Verify layer dimensions
        assert linear.in_features == 768
        assert linear.out_features == 5120
        assert linear.bits == 3

        # Forward pass with reasonable magnitude input
        x = torch.randn(1, 1, 768, dtype=torch.float16, device="mps") * 2.0

        output = linear(x)

        # Assert output is NOT NaN
        nan_count = torch.isnan(output).sum().item()
        inf_count = torch.isinf(output).sum().item()
        assert nan_count == 0, (
            f"Output contains {nan_count} NaN values out of {output.numel()} elements"
        )
        assert inf_count == 0, (
            f"Output contains {inf_count} Inf values out of {output.numel()} elements"
        )

        # Verify output shape
        assert output.shape == (1, 1, 5120), f"Unexpected output shape: {output.shape}"

    def test_q_b_proj_decode_path(self):
        """Test decode path (M<=16) specifically.

        The decode kernel (gemm_trellis_packed_decode) is the primary
        path used during autoregressive generation.
        """
        require_metal()

        from metal_marlin.trellis.linear import TrellisLinear
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = TrellisModelLoader(str(MODEL_PATH))
        weight = loader.load_weight(0, "self_attn.q_b_proj")
        linear = TrellisLinear.from_trellis_weight(weight, device="mps")

        # Test M=1 (single token decode)
        x = torch.randn(1, 768, dtype=torch.float16, device="mps") * 2.0
        output = linear(x)

        assert not torch.isnan(output).any(), "Decode path produces NaN for M=1"
        assert not torch.isinf(output).any(), "Decode path produces Inf for M=1"
        assert output.shape == (1, 5120)

    def test_expansion_vs_contraction(self):
        """Compare expansion layer (q_b_proj) vs contraction layer (o_proj).

        This helps isolate whether the bug is specific to expansion layers.
        """
        require_metal()

        from metal_marlin.trellis.linear import TrellisLinear
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = TrellisModelLoader(str(MODEL_PATH))

        # Load expansion layer: 768 -> 5120
        q_b_proj = loader.load_weight(0, "self_attn.q_b_proj")
        linear_expand = TrellisLinear.from_trellis_weight(q_b_proj, device="mps")

        # Try to load a contraction layer for comparison
        # o_proj is typically out_features -> hidden_size
        linear_contract: TrellisLinear | None = None
        try:
            o_proj = loader.load_weight(0, "self_attn.o_proj")
            linear_contract = TrellisLinear.from_trellis_weight(o_proj, device="mps")
        except (ValueError, KeyError):
            pass

        # Test expansion layer
        x_expand = torch.randn(1, 768, dtype=torch.float16, device="mps") * 2.0
        out_expand = linear_expand(x_expand)
        expand_has_nan = torch.isnan(out_expand).any().item()

        if linear_contract is not None:
            # Test contraction layer
            x_contract = torch.randn(
                1, linear_contract.in_features, dtype=torch.float16, device="mps"
            ) * 2.0
            out_contract = linear_contract(x_contract)
            contract_has_nan = torch.isnan(out_contract).any().item()

            # Report which layers have issues
            if expand_has_nan and not contract_has_nan:
                pytest.fail(
                    f"NaN bug is specific to expansion layers: "
                    f"q_b_proj ({linear_expand.in_features}->{linear_expand.out_features}) has NaN, "
                    f"o_proj ({linear_contract.in_features}->{linear_contract.out_features}) is OK"
                )
            elif expand_has_nan and contract_has_nan:
                pytest.fail("Both expansion and contraction layers produce NaN")
        else:
            if expand_has_nan:
                pytest.fail(
                    f"Expansion layer q_b_proj ({linear_expand.in_features}->"
                    f"{linear_expand.out_features}) produces NaN"
                )

        # Final assertion
        assert not expand_has_nan, "Expansion layer produces NaN"
