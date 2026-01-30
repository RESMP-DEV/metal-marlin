"""Test complete TrellisModel loading and functionality.

Tests verify model loading from pretrained checkpoints, layer composition,
attention projections, forward pass correctness, and memory usage.
"""

from __future__ import annotations

import gc

import psutil
import pytest
import torch

from metal_marlin._compat import HAS_MPS, torch

# Always import config separately since it has fewer dependencies
try:
    from metal_marlin.trellis_config import TrellisModelConfig

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Skip all tests if model classes are not available
try:
    from metal_marlin.trellis_loader import TrellisModelLoader
    from metal_marlin.trellis_model import TrellisDenseMLP, TrellisModel, TrellisMoEMLP

    HAS_TRELLIS_MODEL = True
except ImportError:
    HAS_TRELLIS_MODEL = False

# Model path for testing
MODEL_PATH = "models/GLM-4.7-Flash-EXL3-3bpw/"

requires_trellis_model = pytest.mark.skipif(
    not HAS_TRELLIS_MODEL,
    reason="TrellisModel classes not available",
)

requires_model_files = pytest.mark.skipif(
    not HAS_CONFIG or not HAS_TRELLIS_MODEL or not torch or not torch.backends.mps.is_available(),
    reason="Config, TrellisModel or MPS not available",
)


@pytest.fixture
def model_path():
    """Model path for testing."""
    return MODEL_PATH


@pytest.fixture
def config(model_path):
    """Load model configuration."""
    if not HAS_CONFIG:
        pytest.skip("TrellisModelConfig not available")

    try:
        return TrellisModelConfig.from_pretrained(model_path)
    except Exception as e:
        pytest.skip(f"Failed to load config from {model_path}: {e}")


@pytest.fixture
def model_loader(model_path):
    """Create model loader."""
    if not HAS_TRELLIS_MODEL:
        pytest.skip("TrellisModel not available")

    try:
        return TrellisModelLoader(model_path)
    except Exception as e:
        pytest.skip(f"Failed to create loader for {model_path}: {e}")


@requires_model_files
def test_config_layer_counts(config):
    """Test model configuration has correct layer counts."""
    assert config.num_hidden_layers == 47, f"Expected 47 layers, got {config.num_hidden_layers}"
    assert config.hidden_size == 2048, f"Expected hidden_size=2048, got {config.hidden_size}"
    assert config.num_attention_heads == 32, f"Expected 32 heads, got {config.num_attention_heads}"
    assert config.num_experts == 64, f"Expected 64 experts, got {config.num_experts}"


@requires_model_files
def test_moe_layer_detection(config):
    """Test MoE layer detection - layer 0 is dense, 1-46 are MoE."""
    # Layer 0 should be dense
    assert not config.is_moe_layer(0), "Layer 0 should be dense (first layer)"

    # Layers 1-46 should be MoE
    for i in range(1, 47):
        assert config.is_moe_layer(i), f"Layer {i} should be MoE"

    # Test bounds checking - is_moe_layer doesn't check bounds by design
    assert config.is_moe_layer(47), "Layer 47 (>= first_moe_layer) returns True by design"


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_model_loads_all_layers(model_path, config):
    """Test TrellisModel.from_pretrained() loads all 47 layers."""
    if not HAS_MPS:
        pytest.skip("MPS required for model loading")

    # Load model
    model = TrellisModel.from_pretrained(model_path, device="mps")

    # Verify layer count
    assert len(model.layers) == 47, f"Expected 47 layers, got {len(model.layers)}"

    # Verify embedding and norm are loaded
    assert model.embed_tokens.weight.shape == (config.vocab_size, config.hidden_size)
    assert model.norm.weight.shape == (config.hidden_size,)

    # Verify model is on correct device
    assert next(model.parameters()).device.type == "mps"

    # Clean up
    del model
    torch.mps.empty_cache()
    gc.collect()


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_layer_mlp_types(model_loader, config):
    """Test each layer has correct MLP type (dense for 0, MoE for 1-46)."""
    if not HAS_MPS:
        pytest.skip("MPS required")

    # Load router weights for MoE layers
    router_weights = {}
    if config.num_experts > 1:
        router_weights = model_loader.load_router_weights()

    # Load base weights
    from pathlib import Path

    from safetensors.torch import load_file

    base_weights_path = Path(MODEL_PATH) / "base_weights.safetensors"
    base_weights = load_file(base_weights_path)

    # Test each layer
    for layer_idx in range(config.num_hidden_layers):
        layer = TrellisDecoderLayer.from_loader(
            model_loader, config, layer_idx, router_weights, base_weights, device="mps"
        )

        if layer_idx == 0:
            # Layer 0 should be dense
            assert isinstance(layer.mlp, TrellisDenseMLP), (
                f"Layer {layer_idx} should have dense MLP"
            )
        else:
            # Layers 1-46 should be MoE
            assert isinstance(layer.mlp, TrellisMoEMLP), f"Layer {layer_idx} should have MoE MLP"
            assert layer.mlp.num_experts_per_tok == 8, f"Layer {layer_idx} should use top-8 experts"

        # Clean up layer to save memory
        del layer
        if hasattr(model_loader, "clear_layer_cache"):
            model_loader.clear_layer_cache(layer_idx)


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_attention_projection_shapes(model_loader, config):
    """Test attention projections have correct shapes."""
    if not HAS_MPS:
        pytest.skip("MPS required")

    # Test a few representative layers (0, 23, 46)
    test_layers = [0, 23, 46]

    # Load base weights
    from pathlib import Path

    from safetensors.torch import load_file

    base_weights_path = Path(MODEL_PATH) / "base_weights.safetensors"
    base_weights = load_file(base_weights_path)

    for layer_idx in test_layers:
        layer_weights = model_loader.load_layer(layer_idx)

        # Check attention projection shapes
        prefix = f"model.layers.{layer_idx}.self_attn"

        # KV projections (low-rank)
        kv_a_shape = layer_weights[f"{prefix}.kv_a_proj.weight"].shape
        assert kv_a_shape == (config.kv_lora_rank + config.kv_lora_rank * 2, config.hidden_size), (
            f"Layer {layer_idx} kv_a_proj shape mismatch: {kv_a_shape}"
        )

        kv_b_shape = layer_weights[f"{prefix}.kv_b_proj.weight"].shape
        assert kv_b_shape == (
            config.num_kv_heads * config.head_dim,
            config.kv_lora_rank + config.kv_lora_rank * 2,
        ), f"Layer {layer_idx} kv_b_proj shape mismatch: {kv_b_shape}"

        # Output projection
        o_shape = layer_weights[f"{prefix}.o_proj.weight"].shape
        assert o_shape == (config.hidden_size, config.num_attention_heads * config.head_dim), (
            f"Layer {layer_idx} o_proj shape mismatch: {o_shape}"
        )

        # Query projections (if present)
        q_a_key = f"{prefix}.q_a_proj.weight"
        q_b_key = f"{prefix}.q_b_proj.weight"
        if q_a_key in layer_weights and q_b_key in layer_weights:
            q_a_shape = layer_weights[q_a_key].shape
            assert q_a_shape == (config.q_lora_rank, config.hidden_size), (
                f"Layer {layer_idx} q_a_proj shape mismatch: {q_a_shape}"
            )

            q_b_shape = layer_weights[q_b_key].shape
            assert q_b_shape == (
                config.num_attention_heads * config.head_dim,
                config.q_lora_rank,
            ), f"Layer {layer_idx} q_b_proj shape mismatch: {q_b_shape}"

        # Clean up
        if hasattr(model_loader, "clear_layer_cache"):
            model_loader.clear_layer_cache(layer_idx)


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_forward_pass_shapes(model_path, config):
    """Test forward pass produces output of correct shape."""
    if not HAS_MPS:
        pytest.skip("MPS required")

    # Load model
    model = TrellisModel.from_pretrained(model_path, device="mps", load_in_layers=True)

    # Test inputs
    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="mps")
    position_ids = torch.arange(seq_len, device="mps").unsqueeze(0).expand(batch_size, -1)

    # Forward pass
    with torch.no_grad():
        output = model(input_ids=input_ids, position_ids=position_ids)

    # Check output shape
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )

    # Check for NaNs
    assert not torch.isnan(output).any(), "Output contains NaN values"

    # Check output is not all zeros
    assert output.abs().sum() > 0, "Output is all zeros"

    # Clean up
    del model, input_ids, position_ids, output
    torch.mps.empty_cache()
    gc.collect()


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_memory_usage(model_path, config):
    """Test memory usage is reasonable (~3-4 bytes per parameter)."""
    if not HAS_MPS:
        pytest.skip("MPS required")

    # Clear memory before test
    torch.mps.empty_cache()
    gc.collect()

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Load model
    model = TrellisModel.from_pretrained(model_path, device="mps", load_in_layers=True)

    # Get memory after loading
    peak_memory = process.memory_info().rss
    memory_used_gb = (peak_memory - initial_memory) / (1024**3)

    # Estimate model size (3-4 bytes per parameter for quantized model)
    # GLM-4.7-Flash-EXL3-3bpw should be ~3bpw = 3 bits per weight
    # Account for KV cache, activation memory, etc.
    expected_size_gb = 8  # Reasonable upper bound for 3bpw model

    print(f"Memory used: {memory_used_gb:.2f} GB")
    print(f"Expected upper bound: {expected_size_gb} GB")

    # Memory should be reasonable (allow some overhead)
    assert memory_used_gb < expected_size_gb * 1.5, (
        f"Memory usage too high: {memory_used_gb:.2f} GB > {expected_size_gb * 1.5:.2f} GB"
    )

    # Clean up
    del model
    torch.mps.empty_cache()
    gc.collect()


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_model_with_kv_cache(model_path, config):
    """Test model forward pass with KV cache for generation."""
    if not HAS_MPS:
        pytest.skip("MPS required")

    from metal_marlin.trellis_kv_cache import TrellisKVCache

    # Load model
    model = TrellisModel.from_pretrained(model_path, device="mps", load_in_layers=True)

    # Create KV cache
    kv_cache = TrellisKVCache(
        batch_size=1,
        max_seq_len=512,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        device="mps",
    )

    # Prefill stage
    batch_size = 1
    prefill_seq_len = 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, prefill_seq_len), device="mps")
    position_ids = torch.arange(prefill_seq_len, device="mps").unsqueeze(0)

    with torch.no_grad():
        prefill_output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

    assert prefill_output.shape == (batch_size, prefill_seq_len, config.hidden_size)

    # Generation stage (single token)
    next_input_ids = torch.randint(0, config.vocab_size, (batch_size, 1), device="mps")
    next_position_ids = torch.tensor([[prefill_seq_len]], device="mps")

    with torch.no_grad():
        gen_output = model(
            input_ids=next_input_ids,
            position_ids=next_position_ids,
            kv_cache=kv_cache,
        )

    assert gen_output.shape == (batch_size, 1, config.hidden_size)

    # Check cache was updated
    assert kv_cache.seq_len == prefill_seq_len + 1

    # Clean up
    del model, kv_cache, input_ids, position_ids, next_input_ids, next_position_ids
    del prefill_output, gen_output
    torch.mps.empty_cache()
    gc.collect()


@requires_trellis_model
@requires_model_files
@pytest.mark.slow
def test_model_deterministic(model_path, config):
    """Test model produces deterministic outputs with same input."""
    if not HAS_MPS:
        pytest.skip("MPS required")

    # Load model twice to ensure no state sharing
    model1 = TrellisModel.from_pretrained(model_path, device="mps", load_in_layers=True)
    model2 = TrellisModel.from_pretrained(model_path, device="mps", load_in_layers=True)

    # Set seeds for deterministic behavior
    torch.manual_seed(42)

    # Create test input
    batch_size = 1
    seq_len = 4
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="mps")
    position_ids = torch.arange(seq_len, device="mps").unsqueeze(0).expand(batch_size, -1)

    # Forward pass through both models
    with torch.no_grad():
        torch.manual_seed(42)
        output1 = model1(input_ids=input_ids, position_ids=position_ids)

        torch.manual_seed(42)
        output2 = model2(input_ids=input_ids, position_ids=position_ids)

    # Check outputs are close (allowing for minor floating point differences)
    torch.testing.assert_close(output1.float().cpu(), output2.float().cpu(), atol=1e-3, rtol=1e-3)

    # Clean up
    del model1, model2, input_ids, position_ids, output1, output2
    torch.mps.empty_cache()
    gc.collect()


# Import TrellisDecoderLayer for tests that need it
if HAS_TRELLIS_MODEL:
    from metal_marlin.trellis_model import TrellisDecoderLayer
