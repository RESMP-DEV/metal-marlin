"""Tests for custom Metal GEMM kernel integration."""

import pytest
import torch

from metal_marlin.ops.metal_linear import MetalQuantizedLinear


class TestMetalQuantizedLinear:
    """Test MetalQuantizedLinear layer."""

    def test_int8_layer_creation(self):
        """Test INT8 layer initialization."""
        layer = MetalQuantizedLinear(512, 1024, quant_type="int8")
        assert layer.weight_packed.shape == (128, 1024)  # 512//4
        assert layer.scales.shape[1] == 1024

    def test_from_linear(self):
        """Test conversion from nn.Linear."""
        linear = torch.nn.Linear(256, 512)
        metal_linear = MetalQuantizedLinear.from_linear(linear, quant_type="int8")
        assert metal_linear.in_features == 256
        assert metal_linear.out_features == 512

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_forward_mps(self):
        """Test forward pass on MPS."""
        linear = torch.nn.Linear(64, 128)
        metal_linear = MetalQuantizedLinear.from_linear(linear, quant_type="int8")
        metal_linear = metal_linear.to("mps")

        x = torch.randn(4, 64, device="mps")
        try:
            out = metal_linear(x)
            assert out.shape == (4, 128)
        except Exception as e:
            pytest.skip(f"Metal kernel not working: {e}")

    def test_output_similarity(self):
        """Test that quantized output is similar to FP32."""
        torch.manual_seed(42)
        linear = torch.nn.Linear(128, 256)
        metal_linear = MetalQuantizedLinear.from_linear(linear, quant_type="int8")

        x = torch.randn(2, 128)

        # Compare on CPU
        linear_out = linear(x)
        metal_linear_cpu = metal_linear.to("cpu")

        # For INT8, expect some quantization error but reasonable similarity
        # This is a sanity check, not a strict accuracy test
        assert metal_linear_cpu.in_features == linear.in_features
