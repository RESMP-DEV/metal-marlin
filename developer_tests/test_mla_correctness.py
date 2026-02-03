"""Test MLA projection correctness after refactoring.

Validates that the Metal MLA projection kernels produce results matching
a reference PyTorch implementation, accounting for FP4 quantization error.
"""

import numpy as np
import torch

from metal_marlin.mla_kernels import (
    mla_proj_fp4,
    pack_fp4_weights_mla,
)


def _to_mps_tensor(x):
    """Convert array or tensor to MPS tensor."""
    if isinstance(x, torch.Tensor):
        return x.to('mps')
    return torch.from_numpy(np.asarray(x)).to('mps')


def test_mla_proj_fp4():
    """Test single MLA projection with FP4 quantized weights."""
    batch, seq, hidden_dim = 1, 64, 512
    latent_dim = 128  # Typical kv_lora_rank

    # Input hidden states
    x = torch.randn(batch, seq, hidden_dim, device='mps', dtype=torch.float16)

    # Weight matrix (hidden_dim -> latent_dim)
    weight = torch.randn(hidden_dim, latent_dim, dtype=torch.float16)

    # Reference: FP16 matmul
    ref = x.float() @ weight.to('mps').float()

    # Quantize weights to FP4
    # pack_fp4_weights_mla expects [out_features, in_features], i.e., weight.T
    weight_packed, scales = pack_fp4_weights_mla(weight.T, group_size=64)

    # Metal FP4 projection
    weight_packed_t = _to_mps_tensor(weight_packed)
    scales_t = _to_mps_tensor(scales)

    result = mla_proj_fp4(x, weight_packed_t, scales_t, group_size=64)

    # FP4 has ~0.5-1% relative error, so use relaxed tolerance
    # Check shape matches
    assert result.shape == (batch, seq, latent_dim), (
        f"Shape mismatch: expected {(batch, seq, latent_dim)}, got {result.shape}"
    )

    # Check values are in reasonable range (not NaN/Inf)
    assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    # Relative error check with FP4-appropriate tolerance
    # FP4 quantization can have 5-10% error, so we check for correlation
    ref_mps = ref.half().to('mps')
    result_mps = result.to('mps')

    # Cosine similarity should be high even with quantization error
    ref_flat = ref_mps.reshape(-1).float()
    result_flat = result_mps.reshape(-1).float()

    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), result_flat.unsqueeze(0)
    ).item()

    assert cosine_sim > 0.9, f"Cosine similarity too low: {cosine_sim:.4f}"
    print(f"✓ MLA FP4 projection correctness validated (cosine sim: {cosine_sim:.4f})")


def test_mla_kv_projection_flow():
    """Test the full KV projection flow (hidden -> latent -> KV)."""
    batch, seq = 1, 64
    hidden_dim = 512
    kv_lora_rank = 128
    num_kv_heads = 4
    head_dim = 64

    # Input hidden states
    hidden = torch.randn(batch, seq, hidden_dim, device='mps', dtype=torch.float16)

    # kv_a: hidden_dim -> kv_lora_rank (down-projection)
    w_kv_a = torch.randn(hidden_dim, kv_lora_rank, dtype=torch.float16)

    # kv_b: kv_lora_rank -> num_kv_heads * 2 * head_dim (up-projection)
    # Factor of 2 for K and V
    kv_out_dim = num_kv_heads * 2 * head_dim
    w_kv_b = torch.randn(kv_lora_rank, kv_out_dim, dtype=torch.float16)

    # Reference: FP16 matmuls
    latent_ref = hidden.float() @ w_kv_a.to('mps').float()
    kv_ref = latent_ref @ w_kv_b.to('mps').float()

    # Quantize both weight matrices
    wa_packed, wa_scales = pack_fp4_weights_mla(w_kv_a.T, group_size=64)
    wb_packed, wb_scales = pack_fp4_weights_mla(w_kv_b.T, group_size=64)

    # Metal FP4 projections
    wa_packed_t = _to_mps_tensor(wa_packed)
    wa_scales_t = _to_mps_tensor(wa_scales)
    wb_packed_t = _to_mps_tensor(wb_packed)
    wb_scales_t = _to_mps_tensor(wb_scales)

    latent_result = mla_proj_fp4(hidden, wa_packed_t, wa_scales_t, group_size=64)
    kv_result = mla_proj_fp4(latent_result, wb_packed_t, wb_scales_t, group_size=64)

    # Validate shapes
    assert latent_result.shape == (batch, seq, kv_lora_rank), (
        f"Latent shape mismatch: expected {(batch, seq, kv_lora_rank)}, "
        f"got {latent_result.shape}"
    )
    assert kv_result.shape == (batch, seq, kv_out_dim), (
        f"KV shape mismatch: expected {(batch, seq, kv_out_dim)}, "
        f"got {kv_result.shape}"
    )

    # Check correlation with reference
    kv_ref_mps = kv_ref.half().to('mps')
    ref_flat = kv_ref_mps.reshape(-1).float()
    result_flat = kv_result.reshape(-1).float()

    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), result_flat.unsqueeze(0)
    ).item()

    # Chained projections accumulate error, so tolerance is lower
    assert cosine_sim > 0.85, f"KV flow cosine similarity too low: {cosine_sim:.4f}"
    print(f"✓ MLA KV projection flow validated (cosine sim: {cosine_sim:.4f})")


def test_mla_qkv_projections():
    """Test Q, K, V projections for MLA attention."""
    batch, seq, heads, head_dim = 1, 64, 8, 128
    latent_dim = 512

    # Input: projected latents (as in original test spec)
    x = torch.randn(batch, seq, latent_dim, device='mps', dtype=torch.float16)

    # MLA projection weights
    w_q = torch.randn(latent_dim, heads * head_dim, dtype=torch.float16)
    w_k = torch.randn(latent_dim, heads * head_dim, dtype=torch.float16)
    w_v = torch.randn(latent_dim, heads * head_dim, dtype=torch.float16)

    # Reference implementation
    q_ref = x.float() @ w_q.to('mps').float()
    k_ref = x.float() @ w_k.to('mps').float()
    v_ref = x.float() @ w_v.to('mps').float()

    # Quantize weights
    wq_packed, wq_scales = pack_fp4_weights_mla(w_q.T, group_size=64)
    wk_packed, wk_scales = pack_fp4_weights_mla(w_k.T, group_size=64)
    wv_packed, wv_scales = pack_fp4_weights_mla(w_v.T, group_size=64)

    # Metal implementation
    q = mla_proj_fp4(x, _to_mps_tensor(wq_packed), _to_mps_tensor(wq_scales), group_size=64)
    k = mla_proj_fp4(x, _to_mps_tensor(wk_packed), _to_mps_tensor(wk_scales), group_size=64)
    v = mla_proj_fp4(x, _to_mps_tensor(wv_packed), _to_mps_tensor(wv_scales), group_size=64)

    # Validate Q
    q_cos = torch.nn.functional.cosine_similarity(
        q_ref.half().reshape(-1).float().unsqueeze(0),
        q.reshape(-1).float().unsqueeze(0)
    ).item()
    assert q_cos > 0.9, f"Q cosine similarity too low: {q_cos:.4f}"

    # Validate K
    k_cos = torch.nn.functional.cosine_similarity(
        k_ref.half().reshape(-1).float().unsqueeze(0),
        k.reshape(-1).float().unsqueeze(0)
    ).item()
    assert k_cos > 0.9, f"K cosine similarity too low: {k_cos:.4f}"

    # Validate V
    v_cos = torch.nn.functional.cosine_similarity(
        v_ref.half().reshape(-1).float().unsqueeze(0),
        v.reshape(-1).float().unsqueeze(0)
    ).item()
    assert v_cos > 0.9, f"V cosine similarity too low: {v_cos:.4f}"

    print("✓ MLA Q/K/V projection correctness validated")
    print(f"  Q cosine sim: {q_cos:.4f}")
    print(f"  K cosine sim: {k_cos:.4f}")
    print(f"  V cosine sim: {v_cos:.4f}")


if __name__ == "__main__":
    test_mla_proj_fp4()
    test_mla_kv_projection_flow()
    test_mla_qkv_projections()
