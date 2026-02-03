"""Test utilities for Trellis models.

Provides factory functions to create minimal test models without loading
14GB weights from disk. Use these for unit tests and CI.

Usage:
    from metal_marlin.trellis.testing import create_mini_model

    # Create a tiny model for testing (~50MB vs 14GB)
    model = create_mini_model()

    # Test forward pass
    x = torch.randn(1, 10, model.config.hidden_size).to("mps")
    output = model.model(x)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import TrellisModelConfig
from .layer import TrellisDenseMLP
from .linear import TrellisLinear
from .model import TrellisDecoderLayer, TrellisForCausalLM, TrellisMoEMLP


@dataclass
class MockTrellisWeight:
    """Mock TrellisWeight for testing without loading real model files."""

    packed_indices: torch.Tensor
    scales: torch.Tensor
    su: torch.Tensor
    sv: torch.Tensor
    bits: int
    original_shape: tuple[int, int]


def create_mock_trellis_linear(
    in_features: int,
    out_features: int,
    bits: int = 3,
    device: str = "mps",
) -> TrellisLinear:
    """Create a TrellisLinear with random packed weights for testing.

    Creates realistic packed Trellis weights that dequantize to reasonable
    values (~0.01-0.1 scale). Memory: ~3 bytes per element (3-bit) vs 2 bytes
    for fp16.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Quantization bits (2, 3, or 4).
        device: Target device.

    Returns:
        TrellisLinear with random but valid packed weights.
    """
    TILE_DIM = 16
    # TrellisWeight convention: K = out_features, N = in_features
    # This matches how TrellisLinear and loader expect weights
    tiles_k = (out_features + TILE_DIM - 1) // TILE_DIM
    tiles_n = (in_features + TILE_DIM - 1) // TILE_DIM
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]

    # Groups are along the N (in_features) dimension
    # This matches TrellisWeight convention: scales [n_groups, N]
    n_groups = max(1, (in_features + 127) // 128)

    # Random packed indices (valid values for bit width)
    packed = torch.randint(0, 256, (tiles_k, tiles_n, packed_bytes), dtype=torch.uint8)

    # Scales: [n_groups, N] = [n_groups, in_features]
    # Each group covers elements along the N (in_features) dimension
    scales = torch.rand(n_groups, in_features, dtype=torch.float32) * 0.1 + 0.01

    # Sign flips following TrellisWeight convention:
    # su: Row signs [K] = [out_features] - one sign per output row
    # sv: Column signs [N] = [in_features] - one sign per input column
    su = torch.where(
        torch.rand(out_features) > 0.5,
        torch.ones(out_features),
        -torch.ones(out_features),
    )
    sv = torch.where(
        torch.rand(in_features) > 0.5,
        torch.ones(in_features),
        -torch.ones(in_features),
    )

    # Use from_trellis_weight to create module with correct buffer shapes
    mock_weight = MockTrellisWeight(
        packed_indices=packed,
        scales=scales,
        su=su.float(),
        sv=sv.float(),
        bits=bits,
        original_shape=(out_features, in_features),
    )

    return TrellisLinear.from_trellis_weight(mock_weight, device=device)


def create_mock_dense_mlp(
    hidden_dim: int,
    intermediate_dim: int,
    bits: int = 3,
    device: str = "mps",
) -> TrellisDenseMLP:
    """Create a mock TrellisDenseMLP for testing.

    Args:
        hidden_dim: Input/output dimension.
        intermediate_dim: MLP hidden dimension.
        bits: Quantization bits (2, 3, or 4).
        device: Target device.
    """
    gate_proj = create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    up_proj = create_mock_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    down_proj = create_mock_trellis_linear(intermediate_dim, hidden_dim, bits, device)
    return TrellisDenseMLP(gate_proj, up_proj, down_proj)


def create_mock_moe_mlp(
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    bits: int = 3,
    device: str = "mps",
    eager_buffers: bool = True,
) -> TrellisMoEMLP:
    """Create a mock TrellisMoEMLP for testing fast vs slow path accuracy.

    Creates a small MoE layer with random weights. Default dimensions
    use ~50MB vs 2GB for a real GLM layer.

    Args:
        hidden_dim: Input/output dimension (default 256).
        intermediate_dim: Per-expert hidden dimension (default 512).
        num_experts: Number of routed experts (default 4).
        num_experts_per_tok: Top-k experts per token (default 2).
        bits: Quantization bits (default 3).
        device: Target device.
        eager_buffers: If True, create Metal buffers eagerly and free PyTorch
            tensors. Set False for validation scripts that need both fast and
            slow paths to work (slow path uses PyTorch weights directly).
    """
    router = nn.Linear(hidden_dim, num_experts, bias=False, device=device, dtype=torch.float32)
    nn.init.xavier_uniform_(router.weight)

    experts = [
        create_mock_dense_mlp(hidden_dim, intermediate_dim, bits, device)
        for _ in range(num_experts)
    ]

    shared_expert = create_mock_dense_mlp(hidden_dim, intermediate_dim, bits, device)

    return TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=shared_expert,
        num_experts_per_tok=num_experts_per_tok,
        eager_buffers=eager_buffers,
    )


def create_mini_config(
    hidden_size: int = 256,
    num_hidden_layers: int = 2,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    intermediate_size: int = 512,
    moe_intermediate_size: int = 256,
    vocab_size: int = 1000,
) -> TrellisModelConfig:
    """Create a minimal TrellisModelConfig for testing.

    Default config creates a ~50MB model vs 14GB for GLM-4.7-Flash.

    Args:
        hidden_size: Model hidden dimension (default 256).
        num_hidden_layers: Number of transformer layers (default 2).
        num_experts: MoE expert count (default 4).
        num_experts_per_tok: Top-k experts (default 2).
        intermediate_size: Dense MLP hidden size (default 512).
        moe_intermediate_size: MoE expert hidden size (default 256).
        vocab_size: Vocabulary size (default 1000).
    """
    return TrellisModelConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        # MoE
        num_experts=num_experts,
        num_shared_experts=1,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        first_moe_layer=1,  # Layer 0 dense, layer 1+ MoE
        # Disable MLA (use standard GQA for simplicity)
        kv_lora_rank=None,
        q_lora_rank=None,
        # Defaults
        rope_theta=10000.0,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        quantization_bits=3,
    )


def create_mini_model(
    config: TrellisModelConfig | None = None,
    device: str = "mps",
    seed: int = 42,
) -> TrellisForCausalLM:
    """Create a minimal TrellisForCausalLM for testing.

    Creates a complete model with random weights, suitable for testing
    forward passes, kernel correctness, and generation logic without
    loading 14GB from disk.

    Memory: ~50MB vs 14GB for real GLM-4.7-Flash.

    Args:
        config: Model config (uses create_mini_config() defaults if None).
        device: Target device (default "mps").
        seed: Random seed for reproducibility.

    Returns:
        TrellisForCausalLM with random weights ready for testing.

    Example:
        >>> model = create_mini_model()
        >>> x = torch.randint(0, 1000, (1, 10)).to("mps")
        >>> logits = model(x)
        >>> assert not logits.isnan().any()
    """
    torch.manual_seed(seed)

    if config is None:
        config = create_mini_config()

    model = TrellisForCausalLM(config)

    # Initialize embedding
    nn.init.normal_(model.model.embed_tokens.weight, std=0.02)

    # Initialize LM head
    nn.init.normal_(model.lm_head.weight, std=0.02)

    # Initialize norm
    nn.init.ones_(model.model.norm.weight)

    # Create layers with mock weights
    for layer_idx in range(config.num_hidden_layers):
        layer = _create_mock_decoder_layer(config, layer_idx, device)
        model.model.layers.append(layer)

    return model.to(device)


def _create_mock_decoder_layer(
    config: TrellisModelConfig,
    layer_idx: int,
    device: str,
) -> TrellisDecoderLayer:
    """Create a mock decoder layer with random weights."""

    layer = TrellisDecoderLayer(config, layer_idx, device)

    # Create attention (simplified - not MLA for testing)
    # For now, create placeholders that forward identity
    layer.self_attn = _MockAttention(config, device)

    # Create MLP (dense or MoE based on layer_idx)
    if config.is_moe_layer(layer_idx):
        layer.mlp = create_mock_moe_mlp(
            hidden_dim=config.hidden_size,
            intermediate_dim=config.moe_intermediate_size or config.intermediate_size // 4,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bits=config.quantization_bits,
            device=device,
        )
    else:
        layer.mlp = create_mock_dense_mlp(
            hidden_dim=config.hidden_size,
            intermediate_dim=config.intermediate_size,
            bits=config.quantization_bits,
            device=device,
        )

    # Initialize layernorms
    nn.init.ones_(layer.input_layernorm.weight)
    nn.init.ones_(layer.post_attention_layernorm.weight)

    return layer


class _MockAttention(nn.Module):
    """Simplified attention for testing (no MLA complexity)."""

    def __init__(self, config: TrellisModelConfig, device: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Simple linear for testing - not real attention
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, device=device)
        nn.init.eye_(self.proj.weight)  # Start as identity

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache=None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Simplified: just project (no actual attention computation)
        # This is sufficient for MoE/MLP kernel testing
        return self.proj(hidden_states)


# Alias for backward compatibility
create_mock_trellis_moe_mlp = create_mock_moe_mlp

__all__ = [
    "create_mini_config",
    "create_mini_model",
    "create_mock_dense_mlp",
    "create_mock_moe_mlp",
    "create_mock_trellis_linear",
    "create_mock_trellis_moe_mlp",
]
