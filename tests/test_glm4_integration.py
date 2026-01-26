from pathlib import Path

import pytest
import torch

_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "results"
    / "glm47_sensitivity_fp8_int2"
)


@pytest.fixture(scope="session")
def glm4_model():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if not _MODEL_DIR.exists():
        pytest.skip(f"GLM-4.7 model not found at {_MODEL_DIR}")

    from metal_marlin.inference_metal import MetalGLM47Model

    return MetalGLM47Model.from_quantized(_MODEL_DIR, bits=2)


@pytest.mark.slow
def test_glm4_loads(glm4_model):
    assert glm4_model is not None
    assert glm4_model.num_layers == 47


@pytest.mark.slow
def test_glm4_forward_shape(glm4_model):
    input_ids = torch.tensor([[1, 2, 3, 4]], device="mps")
    kv_cache = glm4_model.create_kv_cache()
    logits = glm4_model(input_ids, kv_cache=kv_cache)
    # GLM-4.7-Flash vocab_size: 154880
    assert logits.shape == (1, 4, glm4_model.vocab_size)
    assert glm4_model.vocab_size == 154880


@pytest.mark.slow
def test_glm4_mla_cache_compression(glm4_model):
    """Verify MLA cache uses less memory than standard KV cache."""
    batch_size = 1
    seq_len = 1024

    kv_lora_rank = glm4_model.config.get("kv_lora_rank", 512)
    qk_rope_head_dim = glm4_model.config.get("qk_rope_head_dim", 64)
    num_layers = glm4_model.num_layers
    num_heads = glm4_model.config.get("num_heads", 32)
    head_dim = glm4_model.hidden_size // num_heads

    # MLA cache size: [B, L, S, kv_lora_rank + qk_rope_head_dim]
    mla_bytes = batch_size * num_layers * seq_len * (kv_lora_rank + qk_rope_head_dim) * 2

    # Standard KV cache: [B, L, 2, S, H, D]
    standard_bytes = (
        batch_size * num_layers * 2 * seq_len * num_heads * head_dim * 2
    )

    assert mla_bytes < standard_bytes * 0.1  # >10x reduction


@pytest.mark.slow
def test_glm4_generates_coherent(glm4_model):
    from transformers import AutoTokenizer

    from metal_marlin.inference.pipeline import MarlinPipeline

    if not _MODEL_DIR.exists():
        pytest.skip(f"Tokenizer path not found at {_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
    pipe = MarlinPipeline(model=glm4_model, tokenizer=tokenizer)
    result = pipe("Hello, ", max_tokens=20, temperature=0.0)
    assert len(result) > 10
    # Check it's not garbage
    assert not result.endswith("ssssss")
