import pytest
import torch

try:
    from metal_marlin.activation_metal import (
        gelu_metal,
        silu_metal,
        swiglu_fused_metal,
    )
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

@pytest.mark.parametrize("shape", [
    (256,),
    (32, 1024),
    (8, 128, 4096),
])
def test_silu_matches_pytorch(shape):
    x = torch.randn(shape, device="mps", dtype=torch.float16)

    metal_result = silu_metal(x)
    torch_result = torch.nn.functional.silu(x)

    torch.testing.assert_close(metal_result, torch_result, rtol=1e-3, atol=1e-4)

@pytest.mark.parametrize("shape", [
    (256,),
    (32, 1024),
    (8, 128, 4096),
])
def test_gelu_matches_pytorch(shape):
    x = torch.randn(shape, device="mps", dtype=torch.float16)

    metal_result = gelu_metal(x)
    torch_result = torch.nn.functional.gelu(x, approximate="tanh")

    torch.testing.assert_close(metal_result, torch_result, rtol=1e-3, atol=1e-4)

def test_swiglu_fused():
    gate = torch.randn(32, 4096, device="mps", dtype=torch.float16)
    up = torch.randn(32, 4096, device="mps", dtype=torch.float16)

    metal_result = swiglu_fused_metal(gate, up)
    torch_result = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(metal_result, torch_result, rtol=1e-3, atol=1e-4)
