import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import json
import shutil
from pathlib import Path
import sys
import os

# Ensure we can import metal_marlin
sys.path.insert(0, str(Path(__file__).parents[1]))

from metal_marlin.trellis.model import TrellisMoEMLP, TrellisDenseMLP
from metal_marlin.trellis.linear import TrellisLinear

def create_dummy_moe():
    router = nn.Linear(64, 4, bias=False, device='cpu', dtype=torch.float16)
    experts = []
    for _ in range(4):
        # Create dummy experts
        gate = MagicMock(spec=TrellisLinear)
        gate.in_features = 64
        gate.out_features = 256
        gate.bits = 4
        gate.packed_indices = torch.zeros(1, dtype=torch.uint8)
        gate.scales = torch.zeros(1, dtype=torch.float16)
        gate.su = torch.zeros(1, dtype=torch.float16)
        gate.sv = torch.zeros(1, dtype=torch.float16)
        gate.grid = torch.zeros(1, dtype=torch.float16)
        
        up = MagicMock(spec=TrellisLinear)
        up.bits = 4
        up.packed_indices = torch.zeros(1, dtype=torch.uint8)
        up.scales = torch.zeros(1, dtype=torch.float16)
        up.su = torch.zeros(1, dtype=torch.float16)
        up.sv = torch.zeros(1, dtype=torch.float16)
        
        down = MagicMock(spec=TrellisLinear)
        down.bits = 4
        down.packed_indices = torch.zeros(1, dtype=torch.uint8)
        down.scales = torch.zeros(1, dtype=torch.float16)
        down.su = torch.zeros(1, dtype=torch.float16)
        down.sv = torch.zeros(1, dtype=torch.float16)
        
        # TrellisDenseMLP expects TrellisLinear instances
        expert = TrellisDenseMLP(gate, up, down)
        experts.append(expert)
        
    shared = experts[0] # Reuse
    
    # Create with eager_buffers=False to avoid Metal calls during init
    with patch('metal_marlin.trellis.model.HAS_METAL', True):
        moe = TrellisMoEMLP(router, experts, shared, 2, eager_buffers=False)
    
    # Patch internals to avoid Metal/MPS dependency during usage
    moe._get_lib = MagicMock()
    moe._get_cached_buffers = MagicMock(return_value=MagicMock())
    moe._get_buffer_pool = MagicMock(return_value=MagicMock())
    
    # Set dimensions manually
    moe.hidden_dim = 64
    moe.intermediate_dim = 256
    moe.bits = 4
    
    return moe

def test_cache_logic():
    # Setup cache dir
    cache_dir = Path.home() / ".cache" / "metal_marlin"
    # Ensure dir exists or clean it
    if cache_dir.exists():
        # Only remove json files to be safe
        for f in cache_dir.glob("kernel_tuning_*.json"):
            f.unlink()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    moe = create_dummy_moe()
    
    # Mock HAS_METAL and dispatch to avoid running actual kernel
    p1 = patch('metal_marlin.trellis.model.HAS_METAL', True)
    p2 = patch('metal_marlin.trellis.model.dispatch_moe_trellis_swiglu')
    
    with p1, p2 as mock_dispatch:
        mock_dispatch.return_value = torch.zeros(4, 64)
        
        # 1. Run tuning
        x = torch.randn(4, 64, dtype=torch.float16)
        moe._auto_tune_kernel(x)
        
        # Check if cache created
        cache_files = list(cache_dir.glob("kernel_tuning_*.json"))
        assert len(cache_files) == 1, "Cache file should be created"
        
        # Verify content
        content = json.loads(cache_files[0].read_text())
        assert "kernel_config" in content
        assert "cache_key" in content
        assert content["cache_key"]["hidden_dim"] == 64
        
        print("✓ First run created cache")
        
        # 2. Reset moe tuning state to simulate new load
        moe._kernel_auto_tuned = False
        moe._kernel_config = {}
        
        # 3. Run tuning again - should load from cache
        # We check if dispatch is NOT called (since tuning runs dispatch for testing)
        mock_dispatch.reset_mock()
        
        moe._auto_tune_kernel(x)
        
        assert not mock_dispatch.called, "Should not run dispatch test when loading from cache"
        assert moe._kernel_auto_tuned
        assert "optimal" in moe._kernel_config
        
        print("✓ Second run used cache")

if __name__ == "__main__":
    try:
        test_cache_logic()
        print("Test PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)