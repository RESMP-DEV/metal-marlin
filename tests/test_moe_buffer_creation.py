"""Tests for MoE buffer creation from CPU tensors."""

import pytest
import torch


@pytest.fixture
def metal_device():
    """Get Metal device."""
    import Metal
    return Metal.MTLCreateSystemDefaultDevice()


@pytest.fixture
def mock_cpu_weights():
    """Create mock CPU weight tensors for MoE."""
    num_experts = 4
    hidden_dim = 256
    intermediate_dim = 512
    bits = 3

    # Packed indices shape: [experts, tiles_k, tiles_n, packed_bytes]
    tiles_k = (hidden_dim + 15) // 16
    tiles_n = (intermediate_dim + 15) // 16
    packed_bytes = 96  # 3-bit

    gate_weights = torch.randint(0, 256, (num_experts, tiles_k, tiles_n, packed_bytes), dtype=torch.uint8)
    up_weights = torch.randint(0, 256, (num_experts, tiles_k, tiles_n, packed_bytes), dtype=torch.uint8)

    # Down has swapped dims
    down_tiles_k = tiles_n
    down_tiles_n = tiles_k
    down_weights = torch.randint(0, 256, (num_experts, down_tiles_k, down_tiles_n, packed_bytes), dtype=torch.uint8)

    # Scales: [experts, n_groups, out_features]
    n_groups = (hidden_dim + 127) // 128
    gate_scales = torch.randn(num_experts, n_groups, intermediate_dim, dtype=torch.float32)
    up_scales = torch.randn(num_experts, n_groups, intermediate_dim, dtype=torch.float32)
    down_scales = torch.randn(num_experts, (intermediate_dim + 127) // 128, hidden_dim, dtype=torch.float32)

    # Sign vectors
    gate_su = torch.randn(num_experts, hidden_dim, dtype=torch.float32)
    gate_sv = torch.randn(num_experts, intermediate_dim, dtype=torch.float32)
    up_su = torch.randn(num_experts, hidden_dim, dtype=torch.float32)
    up_sv = torch.randn(num_experts, intermediate_dim, dtype=torch.float32)
    down_su = torch.randn(num_experts, intermediate_dim, dtype=torch.float32)
    down_sv = torch.randn(num_experts, hidden_dim, dtype=torch.float32)

    # Grid (codebook)
    grid = torch.randn(8, dtype=torch.float32)  # 3-bit = 8 levels

    return {
        "gate_weights": gate_weights,
        "gate_scales": gate_scales,
        "up_weights": up_weights,
        "up_scales": up_scales,
        "down_weights": down_weights,
        "down_scales": down_scales,
        "gate_su": gate_su,
        "gate_sv": gate_sv,
        "up_su": up_su,
        "up_sv": up_sv,
        "down_su": down_su,
        "down_sv": down_sv,
        "grid": grid,
    }


class TestMoEBuffersFromCPU:
    """Tests for create_cached_weight_buffers_from_cpu."""

    def test_creates_buffers_from_cpu(self, metal_device, mock_cpu_weights):
        """Test buffer creation from CPU tensors."""
        from metal_marlin.trellis.moe_dispatch import create_cached_weight_buffers_from_cpu

        buffers = create_cached_weight_buffers_from_cpu(
            device=metal_device,
            **mock_cpu_weights
        )

        assert buffers.gate_weights is not None
        assert buffers.gate_scales is not None
        assert buffers.up_weights is not None
        assert buffers.down_weights is not None
        assert buffers.grid is not None

    def test_rejects_mps_tensors(self, metal_device, mock_cpu_weights):
        """Test that MPS tensors are rejected."""
        from metal_marlin.trellis.moe_dispatch import create_cached_weight_buffers_from_cpu

        # Move one tensor to MPS
        mock_cpu_weights["gate_weights"] = mock_cpu_weights["gate_weights"].to("mps")

        with pytest.raises(ValueError, match="must be on CPU"):
            create_cached_weight_buffers_from_cpu(
                device=metal_device,
                **mock_cpu_weights
            )
