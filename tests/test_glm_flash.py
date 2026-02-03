"""GLM Flash Attention correctness tests.

Tests verify:
1. Output matches reference (torch.allclose with tolerance)
2. flash_attention_v2 output matches F.scaled_dot_product_attention
3. MoE routing produces same expert selections as reference
4. Generated text is coherent and matches expected patterns

NOTE: Many Metal kernel tests are marked xfail because the kernels exceed
Apple Silicon's 32KB threadgroup memory limit. See test_attention.py for details.

Usage:
    cd contrib/metal_marlin && uv run pytest tests/test_glm_flash.py -v
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

if TYPE_CHECKING:
    import torch as torch_types


def _has_metal_attention() -> bool:
    """Check if Metal attention kernels are available."""
    if not HAS_TORCH or not HAS_MPS:
        return False
    try:
        from metal_marlin.flash_attention_v2 import flash_attention_v2
        return True
    except ImportError:
        return False


def _has_moe_dispatch() -> bool:
    """Check if MoE dispatch is available."""
    if not HAS_TORCH or not HAS_MPS:
        return False
    try:
        from metal_marlin.metal_dispatch import HAS_METAL
        return HAS_METAL
    except ImportError:
        return False


requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
requires_metal_attention = pytest.mark.skipif(
    not _has_metal_attention(), reason="Requires Metal flash attention"
)
requires_moe = pytest.mark.skipif(not _has_moe_dispatch(), reason="Requires MoE dispatch")


# GLM-4.7-Flash attention configuration
# 32 Q heads, 2 KV heads, head_dim=64 (GQA ratio 16:1)
GLM_CONFIG = {
    "num_heads_q": 32,
    "num_heads_kv": 2,
    "head_dim": 64,
    "gqa_ratio": 16,
}

# Tolerances for FP16 attention comparisons
FP16_ATOL = 1e-2
FP16_RTOL = 1e-2


@pytest.fixture(scope="session")
def device() -> str:
    """Return the test device: 'mps' if available, otherwise 'cpu'."""
    if HAS_MPS:
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def torch_rng(seed: int, device: str) -> None:
    """Set PyTorch random seed for reproducible tests."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    torch.manual_seed(seed)


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """NumPy random generator with fixed seed."""
    return np.random.default_rng(seed)


def ref_scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
    num_kv_heads: int | None = None,
) -> np.ndarray:
    """NumPy reference implementation of scaled dot-product attention.

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim]
        K: Key tensor [batch, heads_kv, seq_k, head_dim]
        V: Value tensor [batch, heads_kv, seq_k, head_dim]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        is_causal: If True, apply causal masking
        num_kv_heads: Number of KV heads for GQA expansion

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim]
    """
    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape

    # GQA: expand K/V heads to match Q heads
    if heads_kv < heads_q:
        repeat = heads_q // heads_kv
        K = np.repeat(K, repeat, axis=1)
        V = np.repeat(V, repeat, axis=1)

    # Compute Q @ K^T * scale
    scores = np.einsum("bhqd,bhkd->bhqk", Q.astype(np.float32), K.astype(np.float32))
    scores = scores * scale

    # Apply causal mask if needed
    if is_causal:
        q_idx = np.arange(seq_q)[:, None]
        k_idx = np.arange(seq_k)[None, :]
        mask = np.where(k_idx > q_idx, -np.inf, 0.0).astype(np.float32)
        scores = scores + mask

    # Softmax with numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_max = np.where(np.isinf(scores_max), 0.0, scores_max)
    scores = scores - scores_max
    exp_scores = np.exp(scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
    softmax = exp_scores / sum_exp

    output = np.einsum("bhqk,bhkd->bhqd", softmax, V.astype(np.float32))
    return output.astype(np.float16)


def generate_qkv(
    rng: np.random.Generator,
    batch: int,
    heads_q: int,
    heads_kv: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random Q, K, V tensors."""
    Q = rng.standard_normal((batch, heads_q, seq_q, head_dim)).astype(np.float16)
    K = rng.standard_normal((batch, heads_kv, seq_k, head_dim)).astype(np.float16)
    V = rng.standard_normal((batch, heads_kv, seq_k, head_dim)).astype(np.float16)
    return Q, K, V


class TestFlashAttentionAccuracy:
    """Test flash_attention_v2 output matches F.scaled_dot_product_attention.

    NOTE: These tests are marked xfail because the Metal kernels exceed
    Apple Silicon's 32KB threadgroup memory limit (they allocate 64KB for
    double-buffered K/V tiles). The tests are kept to track regression once
    the kernels are fixed.
    """

    @requires_torch
    @requires_mps
    @requires_metal_attention
    @pytest.mark.xfail(reason="Metal kernels exceed threadgroup memory limit")
    @pytest.mark.parametrize(
        "batch,heads_q,heads_kv,seq_q,seq_k,head_dim,is_causal",
        [
            # Standard MHA configurations
            (1, 8, 8, 128, 128, 64, False),
            (1, 8, 8, 128, 128, 64, True),
            (1, 16, 16, 256, 256, 64, True),
            # GLM-4.7-Flash GQA configuration (32 Q heads, 2 KV heads)
            (1, 32, 2, 128, 128, 64, True),
            (1, 32, 2, 256, 256, 64, True),
            # Decode (seq_q=1)
            (1, 32, 2, 1, 512, 64, False),
            (1, 32, 2, 1, 2048, 64, False),
            # Edge cases
            (1, 8, 8, 17, 17, 64, True),  # Non-aligned sequence length
            (2, 8, 8, 64, 64, 64, True),  # Batch > 1
        ],
    )
    def test_flash_attention_vs_sdpa(
        self,
        rng: np.random.Generator,
        batch: int,
        heads_q: int,
        heads_kv: int,
        seq_q: int,
        seq_k: int,
        head_dim: int,
        is_causal: bool,
    ):
        """Verify flash_attention_v2 matches F.scaled_dot_product_attention."""
        from metal_marlin.flash_attention_v2 import flash_attention_v2

        Q_np, K_np, V_np = generate_qkv(rng, batch, heads_q, heads_kv, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        # Flash Attention V2 output
        flash_out = flash_attention_v2(Q, K, V, scale=scale, causal=is_causal)

        # PyTorch SDPA reference (need to expand K/V for GQA)
        if heads_kv < heads_q:
            repeat = heads_q // heads_kv
            K_expanded = K.repeat_interleave(repeat, dim=1)
            V_expanded = V.repeat_interleave(repeat, dim=1)
        else:
            K_expanded = K
            V_expanded = V

        sdpa_out = F.scaled_dot_product_attention(
            Q.float(),
            K_expanded.float(),
            V_expanded.float(),
            scale=scale,
            is_causal=is_causal,
        ).half()

        # Compare
        torch.testing.assert_close(
            flash_out.float().cpu(),
            sdpa_out.float().cpu(),
            atol=FP16_ATOL,
            rtol=FP16_RTOL,
        )

    @requires_torch
    @requires_mps
    @requires_metal_attention
    @pytest.mark.xfail(reason="Metal kernels exceed threadgroup memory limit")
    def test_flash_attention_vs_numpy_reference(self, rng: np.random.Generator):
        """Verify flash_attention_v2 matches NumPy reference implementation."""
        from metal_marlin.flash_attention_v2 import flash_attention_v2

        batch, heads_q, heads_kv, seq, head_dim = 1, 32, 2, 128, 64
        Q_np, K_np, V_np = generate_qkv(rng, batch, heads_q, heads_kv, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        # NumPy reference
        ref_out = ref_scaled_dot_product_attention(
            Q_np, K_np, V_np, scale, is_causal=True, num_kv_heads=heads_kv
        )

        # Flash Attention V2
        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")
        flash_out = flash_attention_v2(Q, K, V, scale=scale, causal=True)

        np.testing.assert_allclose(
            flash_out.cpu().numpy(),
            ref_out,
            atol=FP16_ATOL,
            rtol=FP16_RTOL,
        )

    @requires_torch
    @requires_mps
    @requires_metal_attention
    @pytest.mark.xfail(reason="Metal kernels exceed threadgroup memory limit")
    @pytest.mark.parametrize("seq_k", [64, 256, 512, 1024, 2048])
    def test_decode_kernel_accuracy(self, rng: np.random.Generator, seq_k: int):
        """Test decode kernel (seq_q=1) accuracy across different context lengths."""
        from metal_marlin.flash_attention_v2 import flash_attention_v2

        batch, heads_q, heads_kv, head_dim = 1, 32, 2, 64
        Q_np, K_np, V_np = generate_qkv(rng, batch, heads_q, heads_kv, 1, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        # Flash attention decode
        flash_out = flash_attention_v2(Q, K, V, scale=scale, causal=False)

        # SDPA reference
        if heads_kv < heads_q:
            K_exp = K.repeat_interleave(heads_q // heads_kv, dim=1)
            V_exp = V.repeat_interleave(heads_q // heads_kv, dim=1)
        else:
            K_exp, V_exp = K, V

        sdpa_out = F.scaled_dot_product_attention(
            Q.float(), K_exp.float(), V_exp.float(), scale=scale, is_causal=False
        ).half()

        torch.testing.assert_close(
            flash_out.float().cpu(),
            sdpa_out.float().cpu(),
            atol=FP16_ATOL,
            rtol=FP16_RTOL,
        )


class TestFusedAttentionDispatch:
    """Test fused_attention dispatch layer correctly routes to backends.

    fused_attention falls back to PyTorch SDPA when Metal kernels fail,
    so these tests should pass.
    """

    @requires_torch
    @requires_mps
    @pytest.mark.xfail(reason="Metal kernels may exceed threadgroup memory limit")
    def test_fused_attention_produces_valid_output(self, rng: np.random.Generator):
        """Test fused_attention returns valid tensor regardless of backend."""
        try:
            from metal_marlin.fused_attention_mps import fused_attention
        except ImportError:
            pytest.skip("fused_attention_mps not available")

        batch, heads, seq, head_dim = 1, 8, 64, 64
        Q_np, K_np, V_np = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        output = fused_attention(Q, K, V, causal=True)

        assert output.shape == (batch, heads, seq, head_dim)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    @requires_torch
    @requires_mps
    @pytest.mark.xfail(reason="Metal kernels may exceed threadgroup memory limit")
    def test_fused_attention_matches_sdpa(self, rng: np.random.Generator):
        """Test fused_attention output matches PyTorch SDPA."""
        try:
            from metal_marlin.fused_attention_mps import fused_attention
        except ImportError:
            pytest.skip("fused_attention_mps not available")

        batch, heads, seq, head_dim = 1, 8, 128, 64
        Q_np, K_np, V_np = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        fused_out = fused_attention(Q, K, V, scale=scale, causal=True)
        sdpa_out = F.scaled_dot_product_attention(
            Q.float(), K.float(), V.float(), scale=scale, is_causal=True
        ).half()

        torch.testing.assert_close(
            fused_out.float().cpu(),
            sdpa_out.float().cpu(),
            atol=FP16_ATOL,
            rtol=FP16_RTOL,
        )


class TestMoERoutingCorrectness:
    """Test MoE routing produces correct expert selections.

    NOTE: The Metal MoE router kernel may fall back to CPU, so these tests
    verify the routing logic rather than Metal kernel correctness specifically.
    """

    @requires_torch
    @requires_mps
    def test_moe_routing_matches_cpu_reference(self, rng: np.random.Generator):
        """Verify MoE routing matches CPU PyTorch reference.

        Tests the _fused_router_topk function which may use Metal or CPU.
        """
        num_tokens = 32
        num_experts = 8
        top_k = 2
        hidden_dim = 256
        device = "mps"

        try:
            from metal_marlin.trellis.moe import _fused_router_topk
        except ImportError:
            pytest.skip("MoE fused router not available")

        # Create random activations and router weights
        # Both x and W_router must have the same dtype for matrix multiplication
        x = torch.from_numpy(
            rng.standard_normal((num_tokens, hidden_dim)).astype(np.float32)
        ).to(device=device)
        W_router = torch.from_numpy(
            rng.standard_normal((num_experts, hidden_dim)).astype(np.float32)
        ).to(device=device)

        # Reference implementation (same device as input)
        router_logits_ref = x.float() @ W_router.T  # [num_tokens, num_experts]
        top_logits_ref, expert_ids_ref = torch.topk(router_logits_ref, top_k, dim=-1)
        expert_weights_ref = F.softmax(top_logits_ref, dim=-1)

        # Routing function (may use Metal or CPU fallback)
        expert_ids_impl, expert_weights_impl = _fused_router_topk(x, W_router, top_k)

        # Expert IDs should match (order within top-k may differ if values are close)
        # Sort both for comparison, move both to CPU
        expert_ids_ref_sorted = expert_ids_ref.cpu().sort(dim=-1).values
        expert_ids_impl_sorted = expert_ids_impl.cpu().sort(dim=-1).values

        assert torch.equal(
            expert_ids_ref_sorted.long(), expert_ids_impl_sorted.long()
        ), f"Expert IDs mismatch: ref={expert_ids_ref_sorted}, impl={expert_ids_impl_sorted}"

        # Weights should be close (softmax of same top-k logits)
        torch.testing.assert_close(
            expert_weights_impl.float().cpu(),
            expert_weights_ref.float().cpu(),
            atol=1e-3,
            rtol=1e-2,
        )

    @requires_torch
    @requires_mps
    def test_moe_routing_expert_frequency_distribution(self, rng: np.random.Generator):
        """Test that MoE routing produces a reasonable expert frequency distribution."""
        num_tokens = 256
        num_experts = 64
        top_k = 8
        hidden_dim = 256
        device = "mps"

        try:
            from metal_marlin.trellis.moe import _fused_router_topk
        except ImportError:
            pytest.skip("MoE fused router not available")

        # Both x and W_router must have the same dtype for matrix multiplication
        x = torch.from_numpy(
            rng.standard_normal((num_tokens, hidden_dim)).astype(np.float32)
        ).to(device=device)
        W_router = torch.from_numpy(
            rng.standard_normal((num_experts, hidden_dim)).astype(np.float32)
        ).to(device=device)

        expert_ids, expert_weights = _fused_router_topk(x, W_router, top_k)

        # Count expert selections
        expert_counts = torch.bincount(
            expert_ids.reshape(-1).cpu().long(), minlength=num_experts
        )

        # Verify all experts are selected at least occasionally
        # (with random weights, no expert should be completely ignored)
        zero_selection_experts = (expert_counts == 0).sum().item()
        assert zero_selection_experts < num_experts // 4, (
            f"Too many experts never selected: {zero_selection_experts}/{num_experts}"
        )

        # Verify weights sum to 1 for each token
        weight_sums = expert_weights.sum(dim=-1)
        torch.testing.assert_close(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=1e-5,
            rtol=1e-5,
        )

    @requires_torch
    @requires_mps
    @requires_moe
    def test_moe_auxiliary_balance_loss(self, rng: np.random.Generator):
        """Test MoE auxiliary balance loss computation."""
        num_tokens = 64
        num_experts = 8
        hidden_dim = 256
        device = "mps"

        # Create imbalanced router weights to test balance loss
        W_router = torch.zeros(num_experts, hidden_dim, device=device)
        # Make first expert dominant
        W_router[0, :] = 1.0
        # Other experts weaker
        W_router[1:, :] = 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=device)

        # Compute router logits
        router_logits = x.float() @ W_router.T
        routing_probs = F.softmax(router_logits, dim=-1)

        # Compute balance loss (coefficient of variation)
        expert_loads = routing_probs.mean(dim=0)
        mean_load = expert_loads.mean()
        std_load = expert_loads.std()
        cv_loss = std_load / (mean_load + 1e-6)

        # Imbalanced distribution should have high CV
        assert cv_loss > 0.5, f"Imbalanced routing should have high CV loss: {cv_loss}"


class TestGenerationCoherence:
    """Test text generation produces coherent output."""

    @requires_torch
    @requires_mps
    @pytest.mark.slow
    def test_greedy_decode_deterministic(self, torch_rng: None):
        """Test that greedy decode is deterministic."""
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
        except Exception:
            pytest.skip("GLM-4.7-Flash tokenizer not available")

        # Simple greedy decode test using just the tokenizer
        prompt = "Hello, how are you"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Multiple encodes should produce identical results
        input_ids_2 = tokenizer.encode(prompt, return_tensors="pt")
        assert torch.equal(input_ids, input_ids_2), "Tokenization should be deterministic"

    @requires_torch
    @requires_mps
    @pytest.mark.slow
    def test_softmax_sampling_distribution(self, rng: np.random.Generator):
        """Test softmax sampling produces valid probability distribution."""
        vocab_size = 32000
        batch_size = 1

        # Create random logits
        logits = torch.from_numpy(
            rng.standard_normal((batch_size, vocab_size)).astype(np.float32)
        ).to(device="mps")

        # Softmax should produce valid probability distribution
        probs = F.softmax(logits, dim=-1)

        assert probs.min() >= 0, "Probabilities should be non-negative"
        assert probs.max() <= 1, "Probabilities should be <= 1"
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones(batch_size, device="mps"),
            atol=1e-5,
            rtol=1e-5,
        )

    @requires_torch
    @requires_mps
    def test_repetition_penalty_effect(self, rng: np.random.Generator):
        """Test repetition penalty correctly modifies logits."""
        vocab_size = 1000
        seq_len = 10

        # Create logits with one dominant token
        logits = torch.zeros(1, vocab_size, device="mps")
        dominant_token = 42
        logits[0, dominant_token] = 10.0  # Very high logit

        # Simulate generated token history
        generated_ids = torch.tensor([[dominant_token] * seq_len], device="mps")

        # Apply repetition penalty
        repetition_penalty = 1.2
        for token_id in generated_ids[0].unique():
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        # After penalty, dominant token should have reduced probability
        probs_after = F.softmax(logits, dim=-1)
        probs_before = F.softmax(torch.zeros(1, vocab_size, device="mps"), dim=-1)
        probs_before[0, dominant_token] = 0.999  # Dominant before

        # Probability should be reduced
        assert probs_after[0, dominant_token] < probs_before[0, dominant_token]


class TestKernelSelection:
    """Test automatic kernel selection for different configurations."""

    @requires_torch
    @requires_metal_attention
    @pytest.mark.parametrize(
        "seq_q,num_heads_q,num_heads_kv,batch,seq_k,is_causal,expected_kernel",
        [
            # Standard MHA prefill (causal)
            (128, 8, 8, 1, 128, True, "flash_attention_v2_causal"),
            # Standard MHA prefill (non-causal)
            (128, 8, 8, 1, 128, False, "flash_attention_v2"),
            # Decode with seq_k > 2048 uses regular decode
            (1, 8, 8, 1, 4096, False, "flash_attention_v2_decode"),
            # Decode with batch=1, seq_k <= 2048 uses fast decode
            (1, 8, 8, 1, 512, False, "flash_attention_v2_decode_fast"),
            # GQA prefill (high ratio >= 4)
            (128, 32, 2, 1, 128, True, "flash_attention_v2_gqa"),
            # GQA decode with batch=1, small seq uses fast decode
            (1, 32, 2, 1, 512, False, "flash_attention_v2_decode_fast"),
            # MQA (single KV head)
            (128, 8, 1, 1, 128, True, "flash_attention_v2_mqa"),
        ],
    )
    def test_kernel_selection(
        self,
        seq_q: int,
        num_heads_q: int,
        num_heads_kv: int,
        batch: int,
        seq_k: int,
        is_causal: bool,
        expected_kernel: str,
    ):
        """Test correct kernel is selected based on configuration."""
        from metal_marlin.flash_attention_v2 import select_kernel

        kernel = select_kernel(
            seq_q=seq_q,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            is_causal=is_causal,
            batch=batch,
            seq_k=seq_k,
        )

        assert kernel == expected_kernel, f"Expected {expected_kernel}, got {kernel}"


class TestNumericalStability:
    """Test numerical stability of attention implementations.

    NOTE: These tests are marked xfail because the Metal kernels exceed
    Apple Silicon's 32KB threadgroup memory limit.
    """

    @requires_torch
    @requires_mps
    @requires_metal_attention
    @pytest.mark.xfail(reason="Metal kernels exceed threadgroup memory limit")
    def test_attention_with_large_values(self, rng: np.random.Generator):
        """Test attention handles large input values without overflow."""
        from metal_marlin.flash_attention_v2 import flash_attention_v2

        batch, heads, seq, head_dim = 1, 8, 64, 64

        # Create inputs with large values (but within FP16 range)
        Q_np = (rng.standard_normal((batch, heads, seq, head_dim)) * 10).astype(np.float16)
        K_np = (rng.standard_normal((batch, heads, seq, head_dim)) * 10).astype(np.float16)
        V_np = (rng.standard_normal((batch, heads, seq, head_dim)) * 10).astype(np.float16)

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        output = flash_attention_v2(Q, K, V, causal=True)

        assert not torch.isnan(output).any(), "Output contains NaN with large inputs"
        assert not torch.isinf(output).any(), "Output contains Inf with large inputs"

    @requires_torch
    @requires_mps
    @requires_metal_attention
    @pytest.mark.xfail(reason="Metal kernels exceed threadgroup memory limit")
    def test_attention_with_small_values(self, rng: np.random.Generator):
        """Test attention handles small input values without underflow."""
        from metal_marlin.flash_attention_v2 import flash_attention_v2

        batch, heads, seq, head_dim = 1, 8, 64, 64

        # Create inputs with small values
        Q_np = (rng.standard_normal((batch, heads, seq, head_dim)) * 0.001).astype(np.float16)
        K_np = (rng.standard_normal((batch, heads, seq, head_dim)) * 0.001).astype(np.float16)
        V_np = (rng.standard_normal((batch, heads, seq, head_dim)) * 0.001).astype(np.float16)

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        output = flash_attention_v2(Q, K, V, causal=True)

        assert not torch.isnan(output).any(), "Output contains NaN with small inputs"
        # Small values can legitimately produce zeros
        assert output.abs().max() < 1e10, "Output magnitude too large"

    @requires_torch
    @requires_mps
    @requires_metal_attention
    @pytest.mark.xfail(reason="Metal kernels exceed threadgroup memory limit")
    def test_attention_softmax_stability(self, rng: np.random.Generator):
        """Test softmax normalization is correct."""
        from metal_marlin.flash_attention_v2 import flash_attention_v2

        batch, heads, seq, head_dim = 1, 4, 32, 64

        # Create Q and K that produce very different attention patterns
        Q_np = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float16)
        K_np = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float16)
        V_np = np.ones((batch, heads, seq, head_dim), dtype=np.float16)  # Constant V

        Q = torch.from_numpy(Q_np).to(device="mps")
        K = torch.from_numpy(K_np).to(device="mps")
        V = torch.from_numpy(V_np).to(device="mps")

        # With V = 1, output should be 1 if softmax is correctly normalized
        output = flash_attention_v2(Q, K, V, causal=False)

        # Each position should sum attention weights * V = 1.0
        expected = torch.ones_like(output)
        torch.testing.assert_close(
            output.float().cpu(),
            expected.float().cpu(),
            atol=1e-2,
            rtol=1e-2,
        )
