import math

import pytest
import torch
import torch.nn.functional as F

from metal_marlin.fused_attention_mps import fused_scaled_dot_product_attention


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
class TestAttentionAccuracy:

    @pytest.mark.parametrize("seq_len_q, seq_len_k", [
        (1, 128),      # Decode
        (128, 128),    # Prefill small
        (512, 512),    # Prefill medium
        (1024, 1024),  # Prefill large
    ])
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_attention_matches_torch(self, seq_len_q, seq_len_k, is_causal):
        # Skip invalid configuration for SDPA/Logic
        if seq_len_q != seq_len_k and is_causal:
             # Torch SDPA behavior for mismatched lengths + causal is specific.
             # Usually for decode (1 vs N), we treat it as non-causal in the kernel call
             # because we are attending to all past tokens.
             pytest.skip("Skipping causal=True for decode/mismatched lengths ambiguity")

        batch_size = 1
        num_heads = 8
        head_dim = 64
        dtype = torch.float16
        device = torch.device("mps")

        q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len_k, head_dim, device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(head_dim)

        # Reference (CPU for stability/correctness of reference)
        ref_q = q.cpu().float()
        ref_k = k.cpu().float()
        ref_v = v.cpu().float()

        ref_out = F.scaled_dot_product_attention(
            ref_q, ref_k, ref_v,
            is_causal=is_causal,
            scale=scale
        )

        # Optimized Implementation (MPS)
        out = fused_scaled_dot_product_attention(
            q, k, v,
            causal=is_causal,
            scale=scale
        )

        # Move to CPU for comparison
        out_cpu = out.cpu().float()

        # Check Max Error
        max_diff = (out_cpu - ref_out).abs().max().item()

        print(f"Config: q={seq_len_q}, k={seq_len_k}, causal={is_causal} -> Max Diff: {max_diff}")

        assert max_diff < 1e-3, f"Max diff {max_diff} exceeds 1e-3"
