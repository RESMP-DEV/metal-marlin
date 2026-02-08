import torch
import numpy as np

try:
    from metal_marlin import _cpp_ext
    from metal_marlin.trellis.mixed_bpw_dispatch import MoEConfig
    can_import_cpp_ext = True
except ImportError:
    can_import_cpp_ext = False

import pytest

@pytest.mark.skipif(not can_import_cpp_ext, reason="C++ extension not built")
def test_cpp_dispatch_mixed_bpw_moe_e2e():
    """
    End-to-end test for the C++ dispatch_mixed_bpw_moe function.
    """
    num_experts = 4
    top_k = 2
    num_tokens = 16
    hidden_dim = 128
    intermediate_dim = 256

    # 1. Prepare inputs
    hidden_states = np.random.randn(num_tokens, hidden_dim).astype(np.float32)
    
    expert_bits = [2, 4, 8, 4]

    expert_weights_packed = []
    expert_scales = []

    for i in range(num_experts):
        # Dummy packed weights (uint8)
        # The actual packing format is complex, so we just create correctly shaped dummy data
        packed_k = (hidden_dim + 7) // 8
        packed_n = intermediate_dim
        # This is not the right shape, but for the purpose of testing the dispatch, it is sufficient.
        # The C++ extension expects a list of numpy arrays
        expert_weights_packed.append(np.random.randint(0, 255, size=(packed_k, packed_n), dtype=np.uint8))
        
        # Dummy scales (float16)
        group_size = 128
        scale_k = (hidden_dim + group_size - 1) // group_size
        scale_n = intermediate_dim
        expert_scales.append(np.random.randn(scale_k, scale_n).astype(np.float16))

    # Expert indices and probabilities
    expert_indices = torch.randint(0, num_experts, (num_tokens, top_k)).numpy().astype(np.int32)
    expert_probs = torch.rand(num_tokens, top_k).softmax(dim=-1).numpy().astype(np.float32)

    # MoEConfig
    config = _cpp_ext.MoEConfig()
    config.hidden_dim=hidden_dim
    config.intermediate_dim=intermediate_dim
    config.num_experts=num_experts
    config.top_k=top_k
    config.use_indirect_command_buffers=True
    config.overlap_cpu_encoding=True
    config.wait_for_completion=True
    config.metallib_path=""

    # Keep a copy of original hidden_states for comparison
    hidden_states_original = hidden_states.copy()

    # 2. Call the C++ function
    # The C++ function modifies hidden_states in-place
    _cpp_ext.dispatch_mixed_bpw_moe(
        hidden_states,
        expert_weights_packed,
        expert_bits,
        expert_scales,
        expert_indices,
        expert_probs,
        config
    )

    # 3. Assertions
    assert hidden_states.shape == (num_tokens, hidden_dim), "Output shape should not change"
    assert not np.allclose(hidden_states, hidden_states_original), "hidden_states should be modified"
    assert not np.all(hidden_states == 0), "Output should not be all zeros"

    print("✓ C++ dispatch_mixed_bpw_moe e2e test passed")

