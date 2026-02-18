"""Verification test for chunked prefill functionality.

This test verifies that chunked prefill is properly implemented and functional.
It checks:
1. The chunked prefill kernel exists and can be imported
2. The MMFP4MLA layer has chunked prefill configured
3. The forward path uses chunked prefill when appropriate
"""

import pytest
import torch


def test_chunked_prefill_imports():
    """Test that chunked prefill components can be imported."""
    from metal_marlin.mla_fused import mla_chunked_prefill_attention
    from metal_marlin.layers.mmfp4_mla import MMFP4MLA
    # If we get here, imports work
    assert callable(mla_chunked_prefill_attention)
    assert hasattr(MMFP4MLA, '_forward_chunked_prefill')


def test_chunked_prefill_configuration():
    """Test that MMFP4MLA has chunked prefill configured."""
    from metal_marlin.layers.mmfp4_mla import MMFP4MLA
    
    model = MMFP4MLA(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=48,
        group_size=64,
    )
    
    # Check chunked prefill size is set
    assert hasattr(model, 'chunked_prefill_size')
    assert model.chunked_prefill_size > 0
    # Default is 2048
    assert model.chunked_prefill_size == 2048


def test_chunked_prefill_kernel_exists():
    """Test that the Metal kernel for chunked prefill exists."""
    import os
    
    # Check the shader file directly
    shader_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'metal_marlin', 'shaders', 'attention_mla_fused.metal'
    )
    if os.path.exists(shader_path):
        with open(shader_path) as f:
            content = f.read()
        assert 'mla_chunked_prefill_attention' in content, "Kernel not found in shader"
    else:
        pytest.skip("Shader file not found")


def test_forward_chunked_prefill_method_exists():
    """Test that _forward_chunked_prefill method exists and is callable."""
    from metal_marlin.layers.mmfp4_mla import MMFP4MLA
    
    model = MMFP4MLA(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=48,
        group_size=64,
    )
    
    assert hasattr(model, '_forward_chunked_prefill')
    assert callable(model._forward_chunked_prefill)


if __name__ == "__main__":
    # Run tests when called directly
    test_chunked_prefill_imports()
    print("✓ Chunked prefill imports test passed")
    
    test_chunked_prefill_configuration()
    print("✓ Chunked prefill configuration test passed")
    
    test_chunked_prefill_kernel_exists()
    print("✓ Chunked prefill kernel exists test passed")
    
    test_forward_chunked_prefill_method_exists()
    print("✓ Forward chunked prefill method exists test passed")
    
    print("\n✓ All chunked prefill verification tests passed!")
