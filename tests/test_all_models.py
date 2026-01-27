"""
Comprehensive test suite for all supported model architectures.

This test downloads small/medium variants of each model family and verifies
that layer replacement + generation works correctly.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metal_marlin import replace_linear_layers

# Model variants to test (smallest available for speed)
MODELS_TO_TEST = [
    # Dense models
    ("meta-llama/Llama-3.2-1B", "llama"),
    ("Qwen/Qwen3-4B", "qwen3"),
    ("mistralai/Mistral-7B-v0.1", "mistral"),
    ("microsoft/phi-2", "phi"),

    # MoE models
    ("Qwen/Qwen3-30B-A3B", "qwen3_moe"),
    ("zai-org/GLM-4.7-Flash", "glm4_moe_lite"),
    # ("mistralai/Mixtral-8x7B-v0.1", "mixtral"),  # Large, optional
]


@pytest.mark.parametrize("model_id,expected_type", MODELS_TO_TEST)
@pytest.mark.slow
def test_model_quantization_and_generation(model_id, expected_type):
    """Test that each model can be loaded, quantized, and generate text."""
    # Load
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="mps",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load {model_id}: {e}")

    # Verify model type
    assert model.config.model_type == expected_type

    # Quantize
    stats = replace_linear_layers(model, bits=4, group_size=128)
    assert stats["replaced_count"] > 0, f"No layers replaced in {model_id}"

    # Generate
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer("Hello", return_tensors="pt").input_ids.to("mps")
    output = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    # Verify output
    assert output.shape[1] > input_ids.shape[1], f"No tokens generated for {model_id}"
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(decoded) > 5, f"Output too short for {model_id}: {decoded}"

    # Cleanup
    del model
    torch.mps.empty_cache()
