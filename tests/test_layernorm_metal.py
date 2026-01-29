import pytest
import torch

try:
    from metal_marlin.layernorm_metal import layernorm_metal, rmsnorm_metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

@pytest.mark.parametrize("shape", [
    (32, 1024),
    (8, 128, 4096),
    (4, 256, 7168),  # GLM-4 hidden
])
def test_rmsnorm_matches_pytorch(shape):
    x = torch.randn(shape, device="mps", dtype=torch.float16)
    hidden = shape[-1]
    weight = torch.randn(hidden, device="mps", dtype=torch.float16)

    # Metal implementation
    metal_result = rmsnorm_metal(x, weight, eps=1e-6)

    # PyTorch reference
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + 1e-6)
    torch_result = (x_normed * weight).half()

    torch.testing.assert_close(metal_result, torch_result, rtol=1e-2, atol=1e-3)

def test_layernorm_matches_pytorch():
    x = torch.randn(32, 4096, device="mps", dtype=torch.float16)
    weight = torch.randn(4096, device="mps", dtype=torch.float16)
    bias = torch.randn(4096, device="mps", dtype=torch.float16)

    metal_result = layernorm_metal(x, weight, bias, eps=1e-5)
    torch_result = torch.nn.functional.layer_norm(x, [4096], weight, bias, eps=1e-5)

    torch.testing.assert_close(metal_result, torch_result, rtol=1e-2, atol=1e-3)
