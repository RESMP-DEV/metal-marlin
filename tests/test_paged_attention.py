"""Tests validating parity between Paged Attention and Linear Attention.

Paged Attention: Uses block-table KV storage, decoupling logical token
positions from physical memory layout. Suitable for batched serving with
variable-length sequences.

Linear Attention: Standard scaled dot-product attention with contiguous KV cache.
Simulates the traditional attention computation approach.

This test suite validates that both implementations produce identical outputs
within acceptable numerical tolerance for equivalent inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.paged.attention import paged_attention, paged_attention_v1


def _get_device() -> str:
    """Get the appropriate test device."""
    if HAS_MPS:
        return "mps"
    return "cpu"


@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
class TestPagedLinearAttentionParity:
    """Validate parity between paged and linear attention outputs."""

    @pytest.mark.parametrize("num_seqs", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    @pytest.mark.parametrize("head_dim", [64, 96, 128])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    @pytest.mark.parametrize("num_kv_heads", [None, 4, 8])  # None = same as num_heads
    @pytest.mark.parametrize("block_size", [16, 32])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_paged_v1_matches_linear_attention(
        self,
        num_seqs,
        num_heads,
        head_dim,
        seq_len,
        num_kv_heads,
        block_size,
        dtype,
        seed,
    ):
        """Paged Attention v1 matches Linear Attention for decode.

        Compares outputs of:
        - paged_attention_v1: Block-table based attention
        - F.scaled_dot_product_attention: Standard linear attention

        Validates numerical equivalence for decode workloads (single Q token).
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Only test GQA configurations where num_heads % num_kv_heads == 0
        if num_heads % num_kv_heads != 0:
            pytest.skip("Invalid GQA configuration")

        device = _get_device()
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate input tensors
        # Paged v1 is decode-only: Q shape is [num_seqs, num_heads, head_dim]
        q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch_dtype, device=device)

        # KV tensors: [num_seqs, seq_len, num_kv_heads, head_dim]
        k = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch_dtype, device=device)
        v = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch_dtype, device=device)

        # Create paged KV cache layout
        # v1 format: [num_blocks, block_size, num_heads, head_dim]
        num_blocks = (seq_len + block_size - 1) // block_size
        k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            dtype=torch_dtype, device=device
        )
        v_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            dtype=torch_dtype, device=device
        )

        # Populate KV cache blocks
        for seq_idx in range(num_seqs):
            for token_idx in range(seq_len):
                block_idx = token_idx // block_size
                slot_idx = token_idx % block_size
                k_cache[block_idx, slot_idx] = k[seq_idx, token_idx]
                v_cache[block_idx, slot_idx] = v[seq_idx, token_idx]

        # Create block table: maps logical blocks to physical blocks
        # For simple case, identity mapping
        max_blocks_per_seq = num_blocks
        block_tables = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0).repeat(num_seqs, 1)
        context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

        # Scale factor for attention
        scale = 1.0 / math.sqrt(head_dim)

        # === Paged Attention (v1) ===
        # NumPy reference implementation
        q_np = q.cpu().numpy()
        k_cache_np = k_cache.cpu().numpy()
        v_cache_np = v_cache.cpu().numpy()
        block_tables_np = block_tables.cpu().numpy()
        context_lens_np = context_lens.cpu().numpy()

        paged_output_np = paged_attention_v1(
            query=q_np,
            k_cache=k_cache_np,
            v_cache=v_cache_np,
            block_tables=block_tables_np,
            context_lens=context_lens_np,
            scale=scale,
        )
        paged_output = torch.from_numpy(paged_output_np).to(device=device, dtype=torch_dtype)

        # === Linear Attention (PyTorch SDPA) ===
        # Reshape Q to [num_seqs, num_heads, 1, head_dim] for SDPA
        q_expanded = q.unsqueeze(2)  # [num_seqs, num_heads, 1, head_dim]
        # Reshape K/V to [num_seqs, num_heads, seq_len, head_dim]
        k_reshaped = k.transpose(1, 2)  # [num_seqs, num_kv_heads, seq_len, head_dim]
        v_reshaped = v.transpose(1, 2)  # [num_seqs, num_kv_heads, seq_len, head_dim]

        # GQA expansion: repeat KV heads to match query heads
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
            v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)

        # Linear attention using PyTorch SDPA
        # For decode: single Q token, no causal mask needed
        linear_output = F.scaled_dot_product_attention(
            q_expanded, k_reshaped, v_reshaped,
            is_causal=False,
            scale=scale
        )
        linear_output = linear_output.squeeze(2)  # [num_seqs, num_heads, head_dim]

        # === Compare outputs ===
        # Move to CPU for comparison
        paged_cpu = paged_output.float().cpu()
        linear_cpu = linear_output.float().cpu()

        max_diff = (paged_cpu - linear_cpu).abs().max().item()
        mean_diff = (paged_cpu - linear_cpu).abs().mean().item()

        # Tolerance based on dtype (more lenient for FP16)
        atol = 1e-3 if dtype == "float16" else 1e-5
        rtol = 0.02 if dtype == "float16" else 0.01

        # Assert closeness
        assert torch.allclose(
            paged_cpu, linear_cpu, atol=atol, rtol=rtol
        ), (
            f"Paged Attention v1 does not match Linear Attention\n"
            f"  Config: seqs={num_seqs}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, dim={head_dim}, seq_len={seq_len}\n"
            f"  dtype={dtype}, seed={seed}, block_size={block_size}\n"
            f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}\n"
            f"  atol={atol}, rtol={rtol}"
        )

    @pytest.mark.parametrize("num_seqs", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [4, 8])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    @pytest.mark.parametrize("num_kv_heads", [None, 4])
    @pytest.mark.parametrize("block_size", [16])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_paged_block_pool_matches_linear_attention(
        self,
        num_seqs,
        num_heads,
        head_dim,
        seq_len,
        num_kv_heads,
        block_size,
        dtype,
    ):
        """Paged Attention (block pool) matches Linear Attention.

        Tests the unified block pool API (`paged_attention()`) which
        supports both prefill and decode, comparing against standard
        linear attention.
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Only test GQA configurations where num_heads % num_kv_heads == 0
        if num_heads % num_kv_heads != 0:
            pytest.skip("Invalid GQA configuration")

        # Use consistent seed for deterministic comparison
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = _get_device()
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # Generate input tensors
        # Block pool API supports: [num_seqs, num_heads, seq_len, head_dim]
        q = torch.randn(
            num_seqs, num_heads, seq_len, head_dim,
            dtype=torch_dtype, device=device
        )

        # Create block pool and populate
        max_blocks = (seq_len + block_size - 1) // block_size + 1
        block_pool = torch.zeros(
            max_blocks, 2, block_size, num_kv_heads, head_dim,
            dtype=torch_dtype, device=device
        )

        # Block tables
        max_blocks_per_seq = (seq_len + block_size - 1) // block_size + 1
        block_tables = torch.zeros(
            num_seqs, max_blocks_per_seq,
            dtype=torch.int32, device=device
        )
        context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

        # Populate KV for each sequence
        for seq_idx in range(num_seqs):
            # Generate KV tensors
            k = torch.randn(
                seq_len, num_kv_heads, head_dim,
                dtype=torch_dtype, device=device
            )
            v = torch.randn(
                seq_len, num_kv_heads, head_dim,
                dtype=torch_dtype, device=device
            )

            # Write to block pool
            num_needed_blocks = (seq_len + block_size - 1) // block_size
            for blk in range(num_needed_blocks):
                phys_block = blk  # Identity mapping
                block_tables[seq_idx, blk] = phys_block

                start_idx = blk * block_size
                end_idx = min((blk + 1) * block_size, seq_len)

                for token_idx in range(start_idx, end_idx):
                    slot_idx = token_idx % block_size
                    block_pool[phys_block, 0, slot_idx] = k[token_idx]
                    block_pool[phys_block, 1, slot_idx] = v[token_idx]

        # Scale factor
        scale = 1.0 / math.sqrt(head_dim)

        # === Paged Attention (block pool) ===
        q_np = q.cpu().numpy()
        block_pool_np = block_pool.cpu().numpy()
        block_tables_np = block_tables.cpu().numpy()
        context_lens_np = context_lens.cpu().numpy()

        paged_output_np = paged_attention(
            query=q_np,
            block_pool=block_pool_np,
            block_tables=block_tables_np,
            context_lens=context_lens_np,
            scale=scale,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
        )
        paged_output = torch.from_numpy(paged_output_np).to(device=device, dtype=torch_dtype)

        # === Linear Attention ===
        # Reshape for SDPA: [num_seqs, num_heads, seq_len, head_dim] is already correct
        # Need K/V: [num_seqs, num_heads, seq_len, head_dim]

        # Gather KV from block pool for linear attention
        # Simulating contiguous cache layout
        k_gathered = []
        v_gathered = []
        for seq_idx in range(num_seqs):
            k_seq = []
            v_seq = []
            for blk in range(num_needed_blocks):
                phys_block = int(block_tables_np[seq_idx, blk])
                k_seq.append(block_pool_np[phys_block, 0])
                v_seq.append(block_pool_np[phys_block, 1])
            k_concat = np.concatenate(k_seq, axis=0)[:seq_len]
            v_concat = np.concatenate(v_seq, axis=0)[:seq_len]
            k_gathered.append(k_concat)
            v_gathered.append(v_concat)

        k_np = np.stack(k_gathered, axis=0)  # [num_seqs, seq_len, num_kv_heads, head_dim]
        v_np = np.stack(v_gathered, axis=0)

        k = torch.from_numpy(k_np).to(device=device, dtype=torch_dtype)
        v = torch.from_numpy(v_np).to(device=device, dtype=torch_dtype)

        # Transpose to [num_seqs, num_heads, seq_len, head_dim]
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)

        # GQA expansion
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
            v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)

        # Linear attention
        linear_output = F.scaled_dot_product_attention(
            q, k_reshaped, v_reshaped,
            is_causal=True,  # Prefill requires causal masking
            scale=scale
        )

        # === Compare outputs ===
        paged_cpu = paged_output.float().cpu()
        linear_cpu = linear_output.float().cpu()

        max_diff = (paged_cpu - linear_cpu).abs().max().item()
        mean_diff = (paged_cpu - linear_cpu).abs().mean().item()

        # Tolerance
        atol = 1e-3 if dtype == "float16" else 1e-5
        rtol = 0.02 if dtype == "float16" else 0.01

        assert torch.allclose(
            paged_cpu, linear_cpu, atol=atol, rtol=rtol
        ), (
            f"Paged Attention (block pool) does not match Linear Attention\n"
            f"  Config: seqs={num_seqs}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, dim={head_dim}, seq_len={seq_len}\n"
            f"  dtype={dtype}, block_size={block_size}\n"
            f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}\n"
            f"  atol={atol}, rtol={rtol}"
        )


@pytest.mark.smoke
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [16, 32])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_v1_smoke(num_heads, head_dim, seq_len, num_kv_heads, block_size):
    """Smoke test for paged_attention_v1 parity with linear attention.

    Quick sanity check with single sequence to verify core correctness.
    Marked with @smoke for fast feedback in development.
    """
    if num_heads % num_kv_heads != 0:
        pytest.skip("Invalid GQA configuration")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _get_device()
    dtype = torch.float16

    # Single sequence for smoke test
    num_seqs = 1

    # Generate inputs
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Paged cache
    num_blocks = (seq_len + block_size - 1) // block_size
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    for token_idx in range(seq_len):
        block_idx = token_idx // block_size
        slot_idx = token_idx % block_size
        k_cache[block_idx, slot_idx] = k[0, token_idx]
        v_cache[block_idx, slot_idx] = v[0, token_idx]

    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    scale = 1.0 / math.sqrt(head_dim)

    # Paged attention
    paged_output_np = paged_attention_v1(
        query=q.cpu().numpy(),
        k_cache=k_cache.cpu().numpy(),
        v_cache=v_cache.cpu().numpy(),
        block_tables=block_tables.cpu().numpy(),
        context_lens=context_lens.cpu().numpy(),
        scale=scale,
    )
    paged_output = torch.from_numpy(paged_output_np).to(device=device, dtype=dtype)

    # Linear attention
    q_expanded = q.unsqueeze(2)
    k_reshaped = k.transpose(1, 2)
    v_reshaped = v.transpose(1, 2)

    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
        v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)

    linear_output = F.scaled_dot_product_attention(
        q_expanded, k_reshaped, v_reshaped,
        is_causal=False, scale=scale
    ).squeeze(2)

    # Verify parity
    assert torch.allclose(
        paged_output.float().cpu(),
        linear_output.float().cpu(),
        atol=1e-3, rtol=0.02
    ), f"Smoke test failed: heads={num_heads}, kv_heads={num_kv_heads}, dim={head_dim}, seq_len={seq_len}"


@pytest.mark.smoke
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("block_size", [16])
def test_paged_block_pool_smoke(num_heads, head_dim, seq_len, block_size):
    """Smoke test for paged_attention (block pool) parity.

    Quick validation of block pool API against linear attention.
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _get_device()
    dtype = torch.float16

    num_seqs = 1
    num_kv_heads = num_heads  # No GQA for smoke test

    q = torch.randn(num_seqs, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    # Block pool setup
    max_blocks = (seq_len + block_size - 1) // block_size + 1
    block_pool = torch.zeros(max_blocks, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    max_blocks_per_seq = max_blocks
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    # Populate block pool
    num_needed_blocks = (seq_len + block_size - 1) // block_size
    for blk in range(num_needed_blocks):
        block_tables[0, blk] = blk
        start_idx = blk * block_size
        end_idx = min((blk + 1) * block_size, seq_len)

        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        for token_idx in range(start_idx, end_idx):
            slot_idx = token_idx % block_size
            block_pool[blk, 0, slot_idx] = k[token_idx]
            block_pool[blk, 1, slot_idx] = v[token_idx]

    scale = 1.0 / math.sqrt(head_dim)

    # Paged attention
    paged_output_np = paged_attention(
        query=q.cpu().numpy(),
        block_pool=block_pool.cpu().numpy(),
        block_tables=block_tables.cpu().numpy(),
        context_lens=context_lens.cpu().numpy(),
        scale=scale,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )
    paged_output = torch.from_numpy(paged_output_np).to(device=device, dtype=dtype)

    # Linear attention - gather KV from block pool
    k_gathered = np.concatenate([block_pool.cpu().numpy()[b, 0] for b in range(num_needed_blocks)], axis=0)[:seq_len]
    v_gathered = np.concatenate([block_pool.cpu().numpy()[b, 1] for b in range(num_needed_blocks)], axis=0)[:seq_len]

    k = torch.from_numpy(k_gathered).unsqueeze(0).to(device=device, dtype=dtype)
    v = torch.from_numpy(v_gathered).unsqueeze(0).to(device=device, dtype=dtype)

    k_reshaped = k.transpose(1, 2)
    v_reshaped = v.transpose(1, 2)

    linear_output = F.scaled_dot_product_attention(
        q, k_reshaped, v_reshaped,
        is_causal=True, scale=scale
    )

    # Verify parity
    assert torch.allclose(
        paged_output.float().cpu(),
        linear_output.float().cpu(),
        atol=1e-3, rtol=0.02
    ), f"Block pool smoke test failed: heads={num_heads}, dim={head_dim}, seq_len={seq_len}"


class TestPagedAttentionEdgeCases:
    """Test edge cases for paged vs linear attention parity."""

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_single_token_decode(self):
        """Single token decode (seq_len=1) matches linear attention."""
        torch.manual_seed(123)
        np.random.seed(123)

        device = _get_device()
        dtype = torch.float16

        num_seqs = 1
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64
        seq_len = 1
        block_size = 16

        q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)

        # KV cache with 1 token
        k = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        k_cache = torch.zeros(1, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_cache = torch.zeros(1, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        k_cache[0, 0] = k[0, 0]
        v_cache[0, 0] = v[0, 0]

        block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([1], dtype=torch.int32, device=device)

        scale = 1.0 / math.sqrt(head_dim)

        # Paged
        paged_output = torch.from_numpy(paged_attention_v1(
            query=q.cpu().numpy(),
            k_cache=k_cache.cpu().numpy(),
            v_cache=v_cache.cpu().numpy(),
            block_tables=block_tables.cpu().numpy(),
            context_lens=context_lens.cpu().numpy(),
            scale=scale,
        )).to(device=device, dtype=dtype)

        # Linear
        q_expanded = q.unsqueeze(2)
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)
        linear_output = F.scaled_dot_product_attention(
            q_expanded, k_reshaped, v_reshaped,
            is_causal=False, scale=scale
        ).squeeze(2)

        assert torch.allclose(
            paged_output.float().cpu(),
            linear_output.float().cpu(),
            atol=1e-3, rtol=0.02
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_gqa_expansion_parity(self):
        """GQA expansion matches between paged and linear attention."""
        torch.manual_seed(456)
        np.random.seed(456)

        device = _get_device()
        dtype = torch.float16

        num_seqs = 2
        num_heads = 16
        num_kv_heads = 4  # GQA ratio of 4
        head_dim = 64
        seq_len = 32
        block_size = 16

        q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        num_blocks = (seq_len + block_size - 1) // block_size
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

        for seq_idx in range(num_seqs):
            for token_idx in range(seq_len):
                block_idx = token_idx // block_size
                slot_idx = token_idx % block_size
                k_cache[block_idx, slot_idx] = k[seq_idx, token_idx]
                v_cache[block_idx, slot_idx] = v[seq_idx, token_idx]

        block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0).repeat(num_seqs, 1)
        context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

        scale = 1.0 / math.sqrt(head_dim)

        # Paged
        paged_output = torch.from_numpy(paged_attention_v1(
            query=q.cpu().numpy(),
            k_cache=k_cache.cpu().numpy(),
            v_cache=v_cache.cpu().numpy(),
            block_tables=block_tables.cpu().numpy(),
            context_lens=context_lens.cpu().numpy(),
            scale=scale,
        )).to(device=device, dtype=dtype)

        # Linear with GQA expansion
        q_expanded = q.unsqueeze(2)
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)

        repeat_factor = num_heads // num_kv_heads
        k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
        v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)

        linear_output = F.scaled_dot_product_attention(
            q_expanded, k_reshaped, v_reshaped,
            is_causal=False, scale=scale
        ).squeeze(2)

        assert torch.allclose(
            paged_output.float().cpu(),
            linear_output.float().cpu(),
            atol=1e-3, rtol=0.02
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_partial_block_usage(self):
        """Attention with partially filled last block matches linear."""
        torch.manual_seed(789)
        np.random.seed(789)

        device = _get_device()
        dtype = torch.float16

        num_seqs = 1
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64
        seq_len = 20  # Non-multiple of block_size
        block_size = 16

        q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        num_blocks = (seq_len + block_size - 1) // block_size  # = 2
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

        for token_idx in range(seq_len):
            block_idx = token_idx // block_size
            slot_idx = token_idx % block_size
            k_cache[block_idx, slot_idx] = k[0, token_idx]
            v_cache[block_idx, slot_idx] = v[0, token_idx]

        block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
        context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

        scale = 1.0 / math.sqrt(head_dim)

        # Paged
        paged_output = torch.from_numpy(paged_attention_v1(
            query=q.cpu().numpy(),
            k_cache=k_cache.cpu().numpy(),
            v_cache=v_cache.cpu().numpy(),
            block_tables=block_tables.cpu().numpy(),
            context_lens=context_lens.cpu().numpy(),
            scale=scale,
        )).to(device=device, dtype=dtype)

        # Linear
        q_expanded = q.unsqueeze(2)
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)
        linear_output = F.scaled_dot_product_attention(
            q_expanded, k_reshaped, v_reshaped,
            is_causal=False, scale=scale
        ).squeeze(2)

        assert torch.allclose(
            paged_output.float().cpu(),
            linear_output.float().cpu(),
            atol=1e-3, rtol=0.02
        )
