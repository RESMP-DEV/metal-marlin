import pytest
from metal_marlin._compat import HAS_MLX

# Skip entire module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")

# Import MLX modules only after skip check
if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    from metal_marlin.layers import MarlinLinear
    from metal_marlin.quantize_model import quantize_model


class TestMarlinLinear:
    def test_basic_forward(self):
        layer = MarlinLinear(512, 256, quant_type="fp4")
        x = mx.random.normal((8, 512))
        out = layer(x)
        assert out.shape == (8, 256)

    def test_from_linear(self):
        linear = nn.Linear(512, 256)
        marlin = MarlinLinear.from_linear(linear, quant_type="fp4")

        x = mx.random.normal((8, 512))
        out_fp16 = linear(x)
        out_marlin = marlin(x)

        # FP4 E2M1 has limited precision (16 representable values).
        # With per-group scaling (group_size=32) and K=512, the accumulated
        # quantization error can be significant. Typical errors:
        # - Mean absolute: ~0.05
        # - Max absolute: ~0.20
        # Use tolerances that accommodate this quantization noise.
        assert mx.allclose(out_fp16, out_marlin, rtol=0.3, atol=0.25)

    def test_batched_input(self):
        layer = MarlinLinear(512, 256)
        x = mx.random.normal((4, 8, 512))  # 3D input
        out = layer(x)
        assert out.shape == (4, 8, 256)

    def test_bias(self):
        layer_bias = MarlinLinear(512, 256, bias=True)
        layer_no_bias = MarlinLinear(512, 256, bias=False)

        assert layer_bias.bias is not None
        assert layer_no_bias.bias is None


class TestQuantizeModel:
    def test_quantize_simple_model(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(512, 256)
                self.fc2 = nn.Linear(256, 128)

            def __call__(self, x):
                return self.fc2(mx.relu(self.fc1(x)))

        model = SimpleModel()
        quantize_model(model, quant_type="fp4")

        assert isinstance(model.fc1, MarlinLinear)
        assert isinstance(model.fc2, MarlinLinear)

    def test_skip_layers(self):
        class ModelWithEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(1000, 512)  # Treat as embedding
                self.fc = nn.Linear(512, 256)

        model = ModelWithEmbed()
        quantize_model(model, skip_layers={"embed"})

        assert isinstance(model.embed, nn.Linear)  # Not quantized
        assert isinstance(model.fc, MarlinLinear)  # Quantized
