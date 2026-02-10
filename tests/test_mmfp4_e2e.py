import pytest
import torch
from pathlib import Path

@pytest.fixture(scope="module")
def model():
    from metal_marlin.models.mmfp4_causal_lm import MMFP4ForCausalLM
    model_path = Path(__file__).parent.parent / "models" / "GLM-4.7-Flash-Marlin-MMFP4"
    if not model_path.exists():
        pytest.skip("Model not found")
    return MMFP4ForCausalLM.from_pretrained(str(model_path), device="mps")

@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5, 8, 16])
def test_forward_no_nan(model, seq_len):
    input_ids = torch.arange(seq_len, device="mps").unsqueeze(0) + 1000
    with torch.inference_mode():
        out = model.forward(input_ids, attention_mask=torch.ones_like(input_ids))
    assert not out.logits.isnan().any(), f"NaN at seq_len={seq_len}"

def test_generate_no_cache(model):
    input_ids = torch.tensor([[9703, 11, 1246]], device="mps")  # "Hello, how"
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=5, use_cache=False)
    assert out.shape[1] == 8  # 3 input + 5 generated

def test_generate_with_cache(model):
    input_ids = torch.tensor([[9703, 11, 1246]], device="mps")
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=5, use_cache=True)
    assert out.shape[1] == 8
