import pytest
import torch
from pathlib import Path
from tests.helpers.synthetic_trellis_fixture import (
    create_synthetic_trellis_fixture,
    get_checked_in_synthetic_trellis_fixture_path,
)
from metal_marlin.trellis.lm import TrellisForCausalLM

def test_create_synthetic_trellis_fixture(tmp_path: Path):
    """
    Tests that the synthetic fixture can be created and loaded by TrellisForCausalLM.
    """
    fixture_meta = create_synthetic_trellis_fixture(tmp_path)

    # 1. Check that the path exists and essential files are present
    model_path = Path(fixture_meta.model_path)
    assert model_path.exists()
    assert (model_path / "config.json").exists()
    assert (model_path / "base_weights.safetensors").exists()
    # Check for at least one layer directory
    assert (model_path / "layer_0000").exists()


    # 2. Load the model from the fixture path
    try:
        model = TrellisForCausalLM.from_pretrained(
            fixture_meta.model_path
        )
    except Exception as e:
        pytest.fail(f"Failed to load model from synthetic fixture: {e}")

    # 3. Check model configuration
    assert model.config.vocab_size == fixture_meta.vocab_size
    assert model.config.hidden_size == fixture_meta.hidden_size
    assert model.config.num_hidden_layers == fixture_meta.layer_count

    # 4. Run a forward pass
    input_ids = torch.randint(0, fixture_meta.vocab_size, (1, 10))
    with torch.no_grad():
        # Move input_ids to the same device as the model
        outputs = model(input_ids.to(next(model.parameters()).device))

    # 5. Check output shape
    assert outputs.logits.shape == (1, 10, fixture_meta.vocab_size)


def test_checked_in_synthetic_trellis_fixture_loads() -> None:
    """Checked-in synthetic fixture should load and run a tiny forward pass."""
    fixture_path = get_checked_in_synthetic_trellis_fixture_path()
    if not fixture_path.exists():
        pytest.skip(f"Checked-in fixture missing: {fixture_path}")

    model = TrellisForCausalLM.from_pretrained(str(fixture_path))
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        outputs = model(input_ids.to(next(model.parameters()).device))
    assert outputs.logits.shape == (1, 4, model.config.vocab_size)
