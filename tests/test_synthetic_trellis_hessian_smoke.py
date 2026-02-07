"""End-to-end smoke regression for synthetic Trellis + Hessian runtime paths."""

from __future__ import annotations

import pytest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None


HAS_MPS = bool(
    torch is not None
    and hasattr(torch, "backends")
    and hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)

# For testing on systems without MPS, use CPU
USE_DEVICE = "mps" if HAS_MPS else "cpu"

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch required")


def _import_runtime_deps():
    """Import runtime-heavy modules lazily to keep collection robust."""
    try:
        from metal_marlin.gptq_metal import GPTQMetal
        from metal_marlin.kv_cache import TrellisKVCache
        from metal_marlin.trellis.lm import TrellisForCausalLM
        from metal_marlin.trellis.model import TrellisModel
        from tests.helpers import get_checked_in_synthetic_trellis_fixture_path
    except Exception as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"Runtime dependencies unavailable: {exc}")

    return (
        TrellisForCausalLM,
        TrellisModel,
        TrellisKVCache,
        GPTQMetal,
        get_checked_in_synthetic_trellis_fixture_path,
    )


@pytest.fixture(scope="module")
def synthetic_checkpoint():
    """Use the checked-in synthetic Trellis fixture for regression coverage."""
    _, _, _, _, get_checked_in_synthetic_trellis_fixture_path = _import_runtime_deps()
    fixture_path = get_checked_in_synthetic_trellis_fixture_path()
    if not fixture_path.exists():
        pytest.skip(f"Checked-in synthetic fixture missing: {fixture_path}")

    config_path = fixture_path / "config.json"
    if not config_path.exists():
        pytest.skip(f"Checked-in synthetic fixture config missing: {config_path}")

    import json

    config = json.loads(config_path.read_text())
    return SimpleNamespace(
        model_path=str(fixture_path),
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_hidden_layers"]),
    )


@pytest.mark.smoke
def test_synthetic_trellis_hessian_smoke(synthetic_checkpoint) -> None:
    """Smoke path: fixture build/load, prefill+decode, and Hessian compute."""
    TrellisForCausalLM, TrellisModel, TrellisKVCache, GPTQMetal, _ = _import_runtime_deps()

    torch.manual_seed(2026)
    model_path = synthetic_checkpoint.model_path

    # 1. Load TrellisForCausalLM and TrellisModel
    causal_model = TrellisForCausalLM.from_pretrained(model_path, device=USE_DEVICE)
    base_model = TrellisModel.from_pretrained(model_path, device=USE_DEVICE)
    causal_model.eval()
    base_model.eval()

    assert causal_model.config.hidden_size == synthetic_checkpoint.hidden_size
    assert len(base_model.layers) == synthetic_checkpoint.layer_count

    # 2. Run tiny prefill step
    prefill_len = 4
    input_ids = torch.randint(
        low=0,
        high=synthetic_checkpoint.vocab_size,
        size=(1, prefill_len),
        device=USE_DEVICE,
        dtype=torch.long,
    )

    kv_cache = TrellisKVCache(
        num_layers=causal_model.config.num_hidden_layers,
        batch_size=1,
        max_seq_len=prefill_len + 4,
        kv_lora_rank=causal_model.config.kv_lora_rank,
        qk_rope_head_dim=causal_model.config.qk_rope_head_dim,
        device=USE_DEVICE,
    )

    with torch.no_grad():
        prefill_logits = causal_model(input_ids, kv_cache=kv_cache).logits

    assert prefill_logits.shape == (1, prefill_len, synthetic_checkpoint.vocab_size)
    assert prefill_logits.device.type == USE_DEVICE
    assert kv_cache.seq_len == prefill_len

    # 3. Run tiny decode step
    decode_ids = torch.randint(
        low=0,
        high=synthetic_checkpoint.vocab_size,
        size=(1, 1),
        device=USE_DEVICE,
        dtype=torch.long,
    )
    with torch.no_grad():
        decode_logits = causal_model(decode_ids, kv_cache=kv_cache).logits

    assert decode_logits.shape == (1, 1, synthetic_checkpoint.vocab_size)
    assert decode_logits.device.type == USE_DEVICE
    assert kv_cache.seq_len == prefill_len + 1

    # 4. Run Hessian computation
    gptq = GPTQMetal()
    # Use hidden_size aligned to 32 as required by GPTQMetal
    activations = torch.randn(
        32,
        synthetic_checkpoint.hidden_size,
        device=USE_DEVICE,
        dtype=torch.float16,
    )
    hessian = gptq.compute_hessian(activations, normalize=True)

    # 5. Assert no crashes and validate output shape/device invariants
    assert hessian.shape == (synthetic_checkpoint.hidden_size, synthetic_checkpoint.hidden_size)
    assert hessian.device.type == USE_DEVICE
    assert torch.isfinite(hessian).all()
    
    # Also verify base model forward
    with torch.no_grad():
        hidden_states = base_model(input_ids)
    assert hidden_states.shape == (1, prefill_len, synthetic_checkpoint.hidden_size)
    assert hidden_states.device.type == USE_DEVICE
