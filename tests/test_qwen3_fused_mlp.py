"""Tests for fused Trellis SwiGLU MLP kernel."""

import pytest
import torch
import torch.nn.functional as F
from metal_marlin.trellis.linear import TrellisLinear
from metal_marlin.trellis.layer import TrellisSwiGLUMlp
from metal_marlin.trellis.loader import TrellisWeight
import numpy as np

def create_mock_trellis_weight(in_features, out_features, bits=4):
    """Create a mock TrellisWeight for testing."""
    TILE_DIM = 16
    tiles_k = (in_features + TILE_DIM - 1) // TILE_DIM
    tiles_n = (out_features + TILE_DIM - 1) // TILE_DIM
    packed_bytes = {2: 64, 3: 96, 4: 128, 8: 256}.get(bits, 128)
    
    packed_indices = torch.randint(0, 255, (tiles_k, tiles_n, packed_bytes), dtype=torch.uint8)
    scales = torch.randn(((in_features + 127) // 128, out_features), dtype=torch.float32).abs()
    su = torch.ones(in_features, dtype=torch.float32)
    sv = torch.ones(out_features, dtype=torch.float32)
    
    return TrellisWeight(
        packed_indices=packed_indices,
        scales=scales,
        su=su,
        sv=sv,
        bits=bits,
        original_shape=(out_features, in_features)
    )

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trellis_swiglu_mlp_correctness():
    """Verify that TrellisSwiGLUMlp matches the reference implementation."""
    hidden_size = 256
    intermediate_size = 512
    bits = 4
    device = "mps"
    
    # Create mock weights
    gate_weight = create_mock_trellis_weight(hidden_size, intermediate_size, bits)
    up_weight = create_mock_trellis_weight(hidden_size, intermediate_size, bits)
    down_weight = create_mock_trellis_weight(intermediate_size, hidden_size, bits)
    
    # Create modules
    gate_proj = TrellisLinear.from_trellis_weight(gate_weight, device=device)
    up_proj = TrellisLinear.from_trellis_weight(up_weight, device=device)
    down_proj = TrellisLinear.from_trellis_weight(down_weight, device=device)
    
    fused_mlp = TrellisSwiGLUMlp(gate_proj, up_proj, down_proj).to(device)
    
    # Input
    batch_size = 4
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
    
    # Reference forward pass (using separate TrellisLinear calls)
    with torch.no_grad():
        gate_out = F.silu(gate_proj(x))
        up_out = up_proj(x)
        expected = down_proj(gate_out * up_out)
        
        # Fused forward pass
        actual = fused_mlp(x)
        
    # Compare
    diff = (actual - expected).abs().max().item()
    print(f"Max difference: {diff}")
    
    # Allow for small numerical differences due to fusion/fast_silu
    assert diff < 1e-2

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trellis_swiglu_mlp_decode():
    """Verify TrellisSwiGLUMlp for batch=1 (decode)."""
    hidden_size = 2560 # Qwen3-4B size
    intermediate_size = 9728
    bits = 4
    device = "mps"
    
    gate_proj = TrellisLinear.from_trellis_weight(
        create_mock_trellis_weight(hidden_size, intermediate_size, bits), device=device)
    up_proj = TrellisLinear.from_trellis_weight(
        create_mock_trellis_weight(hidden_size, intermediate_size, bits), device=device)
    down_proj = TrellisLinear.from_trellis_weight(
        create_mock_trellis_weight(intermediate_size, hidden_size, bits), device=device)
    
    fused_mlp = TrellisSwiGLUMlp(gate_proj, up_proj, down_proj).to(device)
    
    x = torch.randn(1, hidden_size, dtype=torch.float16, device=device)
    
    with torch.no_grad():
        gate_out = F.silu(gate_proj(x))
        up_out = up_proj(x)
        expected = down_proj(gate_out * up_out)
        actual = fused_mlp(x)
        
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-2
