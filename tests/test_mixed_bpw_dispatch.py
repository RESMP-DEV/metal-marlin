"""Test mixed bit-width MoE dispatch with C++ integration."""

import numpy as np
import pytest
import torch

from metal_marlin.trellis.mixed_bpw_dispatch import (
    MoEConfig,
    dispatch_mixed_bpw_moe_with_cpp_fallback,
    get_mixed_bpw_stats,
    reset_mixed_bpw_stats,
)


def _has_metal() -> bool:
    try:
        from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS
        return HAS_METAL and HAS_MPS
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="Metal not available")


@pytest.fixture
def mixed_bpw_config():
    """Create a test configuration for mixed bit-width MoE."""
    return MoEConfig(
        num_experts=4,
        num_experts_per_tok=2,
        hidden_dim=256,
        intermediate_dim=192,
        use_mixed_bpw_optimizations=True,
    )


@pytest.fixture
def mixed_bpw_weights():
    """Create mock weights for mixed bit-width experts."""
    num_experts = 4
    hidden_dim = 256
    intermediate_dim = 192
    bits = 4
    group_size = 128
    n_levels = 1 << bits
    tile_size = 16
    packed_bytes_per_tile = (tile_size * tile_size * bits + 7) // 8

    num_tiles_k = (hidden_dim + tile_size - 1) // tile_size
    num_tiles_n = (intermediate_dim + tile_size - 1) // tile_size

    n_groups = (hidden_dim + group_size - 1) // group_size

    device = "mps"
    expert_weights = {}
    expert_scales = {}
    expert_bits = {}

    # Create experts with different bit-widths
    bit_widths = [2, 4, 4, 8]  # Mixed bit-widths

    for i in range(num_experts):
        bw = bit_widths[i]
        expert_bits[i] = bw

        # Adjust packed size based on bit-width
        packed_bytes_per_tile_bpw = (tile_size * tile_size * bw + 7) // 8

        expert_weights[i] = torch.randint(
            0,
            256,
            (num_tiles_k, num_tiles_n, packed_bytes_per_tile_bpw),
            dtype=torch.uint8,
            device=device,
        )

        expert_scales[i] = torch.randn(
            n_groups, intermediate_dim, dtype=torch.float16, device=device
        )

    return {
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "expert_weights": expert_weights,
        "expert_scales": expert_scales,
        "expert_bits": expert_bits,
    }


def test_mixed_bpw_dispatch_basic(mixed_bpw_config, mixed_bpw_weights):
    """Test basic mixed BPW dispatch executes without error."""
    reset_mixed_bpw_stats()

    config = mixed_bpw_config
    num_experts = config.num_experts
    hidden_dim = config.hidden_dim
    intermediate_dim = config.intermediate_dim

    batch_size = 8
    top_k = config.num_experts_per_tok
    device = "mps"

    # Create input activations
    hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device)

    # Create router outputs
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32, device=device)
    router_probs = torch.softmax(router_logits, dim=-1)

    # Sample expert assignments
    expert_probs, expert_indices = torch.topk(router_probs, top_k, dim=-1)
    expert_indices = expert_indices.to(torch.int32)

    # Run dispatch with C++ fallback
    output = dispatch_mixed_bpw_moe_with_cpp_fallback(
        hidden_states=hidden_states,
        expert_weights=mixed_bpw_weights["expert_weights"],
        expert_scales=mixed_bpw_weights["expert_scales"],
        expert_bits=mixed_bpw_weights["expert_bits"],
        router_probs=router_probs,
        expert_indices=expert_indices,
        config=config,
    )

    # Verify output shape and type
    assert output.shape == (batch_size, hidden_dim)
    assert output.dtype == torch.float16
    assert not output.isnan().any()


def test_mixed_bpw_dispatch_cpp_fallback(mixed_bpw_config, mixed_bpw_weights):
    """Test that C++ fallback is properly integrated."""
    reset_mixed_bpw_stats()

    config = mixed_bpw_config
    num_experts = config.num_experts
    hidden_dim = config.hidden_dim
    intermediate_dim = config.intermediate_dim

    batch_size = 16
    top_k = config.num_experts_per_tok
    device = "mps"

    hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device)

    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32, device=device)
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_probs, expert_indices = torch.topk(router_probs, top_k, dim=-1)
    expert_indices = expert_indices.to(torch.int32)

    # Force C++ path by disabling Metal optimizations
    original_use_mixed = config.use_mixed_bpw_optimizations
    config.use_mixed_bpw_optimizations = False

    try:
        # Import C++ extension to verify it's available
        from metal_marlin import _cpp_ext

        assert hasattr(_cpp_ext, "dispatch_mixed_bpw_moe"), (
            "C++ dispatch function not available"
        )
        assert hasattr(_cpp_ext, "BatchDispatchMixedBPW"), (
            "C++ BatchDispatchMixedBPW class not available"
        )
        assert hasattr(_cpp_ext, "MoEConfig"), "C++ MoEConfig class not available"

        # Test BatchDispatchMixedBPW class
        dispatcher = _cpp_ext.BatchDispatchMixedBPW()
        expert_bits_list = [
            mixed_bpw_weights["expert_bits"][i] for i in range(num_experts)
        ]
        dispatcher.add_expert_bits(expert_bits_list)

        active_experts = [0, 1, 2, 3]
        dispatcher.set_active_experts(active_experts)

        # Test batch building
        expert_indices_np = expert_indices.cpu().numpy().astype(np.int32)
        batches = dispatcher.build_batches_for_routing(
            expert_indices_np,
            max_experts_per_batch=64,
        )

        assert len(batches) > 0, "Should create at least one batch"

        # Test command buffer reservation
        dispatcher.reserve_command_buffers([32, 64, 128], 2)

        # Test stats
        stats = dispatcher.stats()
        assert stats.queued_experts == num_experts
        assert stats.routed_experts == len(active_experts)

    finally:
        config.use_mixed_bpw_optimizations = original_use_mixed


def test_mixed_bpw_config_cpp_integration(mixed_bpw_config):
    """Test that Python and C++ MoEConfig can interoperate."""
    from metal_marlin import _cpp_ext

    # Create Python config
    py_config = mixed_bpw_config

    # Create C++ config
    cpp_config = _cpp_ext.MoEConfig()
    cpp_config.hidden_dim = py_config.hidden_dim
    cpp_config.intermediate_dim = py_config.intermediate_dim
    cpp_config.num_experts = py_config.num_experts
    cpp_config.top_k = py_config.num_experts_per_tok

    # Verify fields are set correctly
    assert cpp_config.hidden_dim == 256
    assert cpp_config.intermediate_dim == 192
    assert cpp_config.num_experts == 4
    assert cpp_config.top_k == 2

    # Test optimization flags
    cpp_config.use_indirect_command_buffers = True
    cpp_config.overlap_cpu_encoding = True
    cpp_config.wait_for_completion = True
    cpp_config.max_experts_per_batch = 64
    cpp_config.command_buffers_per_batch_size = 2

    assert cpp_config.use_indirect_command_buffers is True
    assert cpp_config.overlap_cpu_encoding is True
    assert cpp_config.wait_for_completion is True


def test_mixed_bpw_stats_tracking(mixed_bpw_config, mixed_bpw_weights):
    """Test that statistics are properly tracked."""
    reset_mixed_bpw_stats()

    config = mixed_bpw_config
    num_experts = config.num_experts
    hidden_dim = config.hidden_dim

    batch_size = 8
    top_k = config.num_experts_per_tok
    device = "mps"

    hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device)

    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32, device=device)
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_probs, expert_indices = torch.topk(router_probs, top_k, dim=-1)
    expert_indices = expert_indices.to(torch.int32)

    # Run dispatch
    output = dispatch_mixed_bpw_moe_with_cpp_fallback(
        hidden_states=hidden_states,
        expert_weights=mixed_bpw_weights["expert_weights"],
        expert_scales=mixed_bpw_weights["expert_scales"],
        expert_bits=mixed_bpw_weights["expert_bits"],
        router_probs=router_probs,
        expert_indices=expert_indices,
        config=config,
    )

    # Check stats
    stats = get_mixed_bpw_stats()
    assert stats.total_dispatches >= 1
    assert stats.tokens_processed >= batch_size
    assert stats.experts_activated >= batch_size * top_k


def test_mixed_bpw_bit_width_grouping(mixed_bpw_config, mixed_bpw_weights):
    """Test that experts are correctly grouped by bit-width."""
    from metal_marlin import _cpp_ext

    config = mixed_bpw_config
    num_experts = config.num_experts

    # Create dispatcher with mixed bit-widths
    dispatcher = _cpp_ext.BatchDispatchMixedBPW()

    expert_bits_list = [mixed_bpw_weights["expert_bits"][i] for i in range(num_experts)]
    # Should be [2, 4, 4, 8]
    assert expert_bits_list == [2, 4, 4, 8]

    dispatcher.add_expert_bits(expert_bits_list)

    # Build batches without routing info (just group by bit-width)
    batches = dispatcher.build_batches()

    # Should create 3 batches (one per unique bit-width)
    assert len(batches) == 3

    # Check that each batch has consistent bit-width
    for batch in batches:
        assert batch.bit_width > 0
        assert batch.token_count >= 0
        assert len(batch.expert_ids) > 0
        assert len(batch.expert_ids) == len(batch.expert_token_counts)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
