"""Test that packed format is preserved through the entire pipeline."""

from pathlib import Path

import pytest
import torch

# Skip if model not available
MODEL_PATH = Path(__file__).parents[1] / "models" / "GLM-4.7-Flash-Trellis-3bpw"
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")


def test_weight_stays_packed():
    """Verify weights stay in packed format after loading."""
    from metal_marlin.trellis_loader import TrellisModelLoader

    loader = TrellisModelLoader(str(MODEL_PATH))
    layer = loader.load_layer(0)

    # Find a weight in the layer (e.g., mlp expert weight)
    weight_name = next((k for k in layer.keys() if "gate_proj" in k or "up_proj" in k), None)
    assert weight_name is not None, "No suitable weight found in layer"
    weight = layer[weight_name]

    # Should be uint8 packed, not int16 unpacked
    assert weight.packed_indices.dtype == torch.uint8
    # Should be 96 bytes per tile for 3-bit, not 512
    bytes_per_tile = weight.packed_indices.shape[-1]
    assert bytes_per_tile == 96, f"Expected 96 bytes/tile, got {bytes_per_tile}"


def test_model_memory_efficiency():
    """Verify model uses ~disk size memory, not 5x more."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        baseline = torch.mps.current_allocated_memory()

    from metal_marlin.trellis_lm import TrellisForCausalLM
    model = TrellisForCausalLM.from_pretrained(str(MODEL_PATH), device="mps")

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        used = torch.mps.current_allocated_memory() - baseline

    disk_size = sum(f.stat().st_size for f in MODEL_PATH.rglob("*.safetensors"))

    # Memory should be within 50% of disk size (allowing for FP32 scales, etc)
    efficiency = disk_size / used
    assert efficiency > 0.5, f"Memory efficiency {efficiency:.1%} too low"


def test_forward_pass_no_memory_explosion():
    """Verify forward pass doesn't cache FP16 weights."""
    from metal_marlin.trellis_lm import TrellisForCausalLM
    model = TrellisForCausalLM.from_pretrained(str(MODEL_PATH), device="mps")

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        before = torch.mps.current_allocated_memory()

    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="mps")
        _ = model(input_ids)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        after = torch.mps.current_allocated_memory()

    # Forward should add minimal memory (activations only, not cached weights)
    increase_gb = (after - before) / 1e9
    assert increase_gb < 5.0, f"Forward added {increase_gb:.1f} GB (likely caching weights)"
