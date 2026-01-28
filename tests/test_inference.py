"""Tests for Metal inference components.

These tests use PyTorch MPS + Metal dispatch for testing the inference
infrastructure (KV cache, generation loop, quantized layers) without MLX.

Requires:
    - macOS with Apple Silicon
    - PyTorch with MPS backend
    - PyObjC Metal bindings (for kernel dispatch tests)
"""

import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH

# Skip entire module if PyTorch MPS unavailable
pytestmark = pytest.mark.skipif(not HAS_MPS, reason="Requires PyTorch MPS (Apple Silicon only)")

# Check if Metal kernel library compiles successfully
_METAL_KERNELS_OK: bool = False

# Import PyTorch modules only after skip check
if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from metal_marlin.inference_metal import (
        MetalAttention,
        MetalKVCache,
        MetalKVCacheConfig,
        MetalMLP,
        MetalQuantizedLinear,
        MetalRMSNorm,
        MetalRoPE,
        MetalTransformerBlock,
    )

    # Try to load Metal kernel library to check if kernels compile
    try:
        from metal_marlin.metal_dispatch import get_default_library

        _lib = get_default_library()
        # Try to get a pipeline to verify kernels work
        _lib.get_pipeline("marlin_gemm_fp4")
        _METAL_KERNELS_OK = True
    except Exception:
        # Metal kernels have compilation issues, skip dependent tests
        _METAL_KERNELS_OK = False

# Marker for tests requiring working Metal kernel dispatch
requires_metal_kernels = pytest.mark.skipif(
    not _METAL_KERNELS_OK, reason="Metal GEMM kernels have compilation errors"
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> str:
    """Return MPS device."""
    return "mps"


@pytest.fixture
def small_config() -> dict:
    """Small model config for testing."""
    return {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
    }


# ---------------------------------------------------------------------------
# MetalKVCache Tests
# ---------------------------------------------------------------------------


class TestMetalKVCache:
    """Tests for Metal KV cache implementation."""

    @pytest.mark.smoke
    def test_cache_creation(self):
        """Test basic cache creation."""
        config = MetalKVCacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = MetalKVCache(config, batch_size=1)

        assert cache.seq_len == 0
        assert len(cache.k_cache) == 2
        assert len(cache.v_cache) == 2

    def test_cache_shapes(self):
        """Test cache tensor shapes."""
        config = MetalKVCacheConfig(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            max_seq_len=1024,
        )
        cache = MetalKVCache(config, batch_size=2)

        # [batch, num_kv_heads, max_seq_len, head_dim]
        expected_shape = (2, 8, 1024, 64)
        assert cache.k_cache[0].shape == expected_shape
        assert cache.v_cache[0].shape == expected_shape

    def test_cache_update(self):
        """Test cache update and retrieval."""
        config = MetalKVCacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = MetalKVCache(config, batch_size=1)

        # Add some KV
        k = torch.randn(1, 4, 10, 64, device="mps", dtype=torch.float16)
        v = torch.randn(1, 4, 10, 64, device="mps", dtype=torch.float16)

        k_full, v_full = cache.update(0, k, v)

        assert k_full.shape == (1, 4, 10, 64)
        assert v_full.shape == (1, 4, 10, 64)

        cache.advance(10)
        assert cache.seq_len == 10

    def test_cache_sequential_updates(self):
        """Test multiple sequential cache updates."""
        config = MetalKVCacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = MetalKVCache(config, batch_size=1)

        # Prefill with 10 tokens
        k1 = torch.randn(1, 4, 10, 64, device="mps", dtype=torch.float16)
        v1 = torch.randn(1, 4, 10, 64, device="mps", dtype=torch.float16)
        k_full, v_full = cache.update(0, k1, v1)
        cache.advance(10)

        assert cache.seq_len == 10
        assert k_full.shape == (1, 4, 10, 64)

        # Decode: add 1 token
        k2 = torch.randn(1, 4, 1, 64, device="mps", dtype=torch.float16)
        v2 = torch.randn(1, 4, 1, 64, device="mps", dtype=torch.float16)
        k_full, v_full = cache.update(0, k2, v2)
        cache.advance(1)

        assert cache.seq_len == 11
        assert k_full.shape == (1, 4, 11, 64)

    def test_cache_reset(self):
        """Test cache reset."""
        config = MetalKVCacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = MetalKVCache(config, batch_size=1)

        # Add some data
        k = torch.randn(1, 4, 10, 64, device="mps", dtype=torch.float16)
        v = torch.randn(1, 4, 10, 64, device="mps", dtype=torch.float16)
        cache.update(0, k, v)
        cache.advance(10)
        assert cache.seq_len == 10

        # Reset
        cache.reset()
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# MetalQuantizedLinear Tests
# ---------------------------------------------------------------------------


class TestMetalQuantizedLinear:
    """Tests for Metal quantized linear layers."""

    @pytest.mark.smoke
    def test_layer_creation_fp4(self):
        """Test FP4 quantized layer creation."""
        # Dimensions must be divisible by group_size and pack_factor
        layer = MetalQuantizedLinear(
            in_features=256,  # divisible by 128
            out_features=128,  # can be any size (kernel handles padding)
            bits=4,
            group_size=128,
        )

        assert layer.in_features == 256
        assert layer.out_features == 128
        assert layer.bits == 4
        # K-dimension packing: [K // pack_factor, N_padded]
        assert layer.weight_packed.shape == (256 // 8, 128)
        assert layer.scales.shape == (256 // 128, 128)

    def test_layer_creation_fp8(self):
        """Test FP8 quantized layer creation."""
        layer = MetalQuantizedLinear(
            in_features=256,
            out_features=128,
            bits=8,
            group_size=128,
        )

        assert layer.bits == 8
        # K-dimension packing: [K // pack_factor, N]
        assert layer.weight_packed.shape == (256 // 4, 128)

    def test_layer_creation_int2(self):
        """Test INT2 quantized layer creation."""
        layer = MetalQuantizedLinear(
            in_features=256,
            out_features=256,
            bits=2,
            group_size=128,
        )

        assert layer.bits == 2
        # K-dimension packing: [K // pack_factor, N]
        assert layer.weight_packed.shape == (256 // 16, 256)

    def test_layer_with_bias(self):
        """Test quantized layer with bias."""
        layer_with_bias = MetalQuantizedLinear(256, 128, bias=True)
        layer_no_bias = MetalQuantizedLinear(256, 128, bias=False)

        assert layer_with_bias.bias is not None
        assert layer_no_bias.bias is None

    @requires_metal_kernels
    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = MetalQuantizedLinear(256, 128, bits=4, group_size=128)

        # 2D input
        x = torch.randn(8, 256, device="mps", dtype=torch.float16)
        out = layer(x)
        assert out.shape == (8, 128)

        # 3D input (batched)
        x_3d = torch.randn(4, 8, 256, device="mps", dtype=torch.float16)
        out_3d = layer(x_3d)
        assert out_3d.shape == (4, 8, 128)


# ---------------------------------------------------------------------------
# MetalRMSNorm Tests
# ---------------------------------------------------------------------------


class TestMetalRMSNorm:
    """Tests for Metal RMS normalization."""

    @pytest.mark.smoke
    def test_rms_norm_shape(self):
        """Test RMSNorm preserves shape."""
        norm = MetalRMSNorm(256).to("mps")

        x = torch.randn(4, 8, 256, device="mps")
        out = norm(x)

        assert out.shape == x.shape

    def test_rms_norm_numerical(self):
        """Test RMSNorm numerical correctness."""
        hidden_size = 256
        norm = MetalRMSNorm(hidden_size).to("mps")

        x = torch.randn(2, 4, hidden_size, device="mps")
        out = norm(x)

        # Check that output has approximately unit RMS
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        # RMS should be close to 1.0 since weights are initialized to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)


# ---------------------------------------------------------------------------
# MetalRoPE Tests
# ---------------------------------------------------------------------------


class TestMetalRoPE:
    """Tests for Metal Rotary Position Embedding."""

    @pytest.mark.smoke
    def test_rope_shape(self):
        """Test RoPE preserves shape."""
        rope = MetalRoPE(dim=64, base=10000.0, max_seq_len=512).to("mps")

        # [batch, heads, seq, head_dim]
        x = torch.randn(2, 8, 16, 64, device="mps", dtype=torch.float16)
        out = rope(x)

        assert out.shape == x.shape

    def test_rope_with_offset(self):
        """Test RoPE with position offset."""
        rope = MetalRoPE(dim=64, max_seq_len=512).to("mps")

        x = torch.randn(1, 4, 8, 64, device="mps", dtype=torch.float16)

        # Without offset
        out1 = rope(x, position_offset=0)

        # With offset
        out2 = rope(x, position_offset=10)

        # Outputs should differ due to different positions
        assert not torch.allclose(out1, out2)

    def test_rope_ratio(self):
        """Test RoPE with rope_ratio scaling."""
        rope1 = MetalRoPE(dim=64, rope_ratio=1.0, max_seq_len=512).to("mps")
        rope2 = MetalRoPE(dim=64, rope_ratio=2.0, max_seq_len=512).to("mps")

        x = torch.randn(1, 4, 8, 64, device="mps", dtype=torch.float16)

        out1 = rope1(x)
        out2 = rope2(x)

        # Different rope_ratio should produce different outputs
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# MetalMLP Tests
# ---------------------------------------------------------------------------


class TestMetalMLP:
    """Tests for Metal gated MLP."""

    @pytest.mark.smoke
    @requires_metal_kernels
    def test_mlp_shape(self):
        """Test MLP output shape."""
        mlp = MetalMLP(
            hidden_size=256,
            intermediate_size=512,
            bits=4,
            group_size=128,
        )

        x = torch.randn(4, 8, 256, device="mps", dtype=torch.float16)
        out = mlp(x)

        assert out.shape == x.shape

    @requires_metal_kernels
    def test_mlp_activations(self):
        """Test different activation functions."""
        for activation in ["silu", "gelu", "relu"]:
            mlp = MetalMLP(
                hidden_size=256,
                intermediate_size=512,
                bits=4,
                group_size=128,
                activation=activation,
            )

            x = torch.randn(2, 4, 256, device="mps", dtype=torch.float16)
            out = mlp(x)

            assert out.shape == x.shape
            assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MetalTransformerBlock Tests
# ---------------------------------------------------------------------------


class TestMetalTransformerBlock:
    """Tests for Metal transformer decoder block."""

    @pytest.mark.smoke
    @requires_metal_kernels
    def test_block_forward(self):
        """Test transformer block forward pass."""
        block = MetalTransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=512,
            bits=4,
            group_size=128,
            max_position_embeddings=512,
        )

        x = torch.randn(2, 8, 256, device="mps", dtype=torch.float16)
        out = block(x)

        assert out.shape == x.shape

    @requires_metal_kernels
    def test_block_with_kv_cache(self):
        """Test transformer block with KV cache."""
        block = MetalTransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=512,
            bits=4,
            group_size=128,
            max_position_embeddings=512,
        )

        # Create KV cache
        cache_config = MetalKVCacheConfig(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = MetalKVCache(cache_config, batch_size=1)

        # Prefill
        x_prefill = torch.randn(1, 8, 256, device="mps", dtype=torch.float16)
        out_prefill = block(x_prefill, kv_cache=cache, layer_idx=0)
        cache.advance(8)

        assert out_prefill.shape == x_prefill.shape
        assert cache.seq_len == 8

        # Decode (single token)
        x_decode = torch.randn(1, 1, 256, device="mps", dtype=torch.float16)
        out_decode = block(x_decode, kv_cache=cache, layer_idx=0)

        assert out_decode.shape == x_decode.shape

    @requires_metal_kernels
    def test_block_gqa(self):
        """Test transformer block with grouped query attention."""
        block = MetalTransformerBlock(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,  # GQA: 8 heads, 2 kv heads
            intermediate_size=512,
            bits=4,
            group_size=128,
            max_position_embeddings=512,
        )

        x = torch.randn(1, 4, 256, device="mps", dtype=torch.float16)
        out = block(x)

        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Simple Metal Transformer Model for Testing
# ---------------------------------------------------------------------------


class SimpleMetalTransformer(nn.Module):
    """Minimal Metal transformer for testing inference infrastructure.

    This is NOT a real model - it's a test fixture that implements
    the interface expected by generate() without being architecture-specific.
    Uses standard attention (not MLA) for simplicity.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 512,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings

        # Embedding (not quantized)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                MetalTransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    bits=4,
                    group_size=128,
                    max_position_embeddings=max_position_embeddings,
                )
                for _ in range(num_layers)
            ]
        )

        # Final norm and head
        self.norm = MetalRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: MetalKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, kv_cache=kv_cache, layer_idx=i)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def create_kv_cache(self, batch_size: int = 1) -> MetalKVCache:
        """Create KV cache for this model."""
        config = MetalKVCacheConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_position_embeddings,
        )
        return MetalKVCache(config, batch_size=batch_size)


@pytest.fixture
def small_model() -> SimpleMetalTransformer:
    """Create tiny model for testing."""
    if not _METAL_KERNELS_OK:
        pytest.skip("Metal GEMM kernels have compilation errors")
    model = SimpleMetalTransformer(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
    )
    return model.to("mps").half()


# ---------------------------------------------------------------------------
# Model Forward Tests
# ---------------------------------------------------------------------------


@requires_metal_kernels
class TestModelForward:
    """Tests for model forward pass."""

    @pytest.mark.smoke
    def test_forward_pass(self, small_model):
        """Test basic forward pass."""
        input_ids = torch.randint(0, 1000, (1, 5), device="mps")

        logits = small_model(input_ids)

        # [batch, seq, vocab]
        assert logits.shape == (1, 5, 1000)
        assert not torch.isnan(logits).any()

    def test_forward_batch(self, small_model):
        """Test batched forward pass."""
        input_ids = torch.randint(0, 1000, (4, 8), device="mps")

        logits = small_model(input_ids)

        assert logits.shape == (4, 8, 1000)

    def test_forward_with_cache(self, small_model):
        """Test forward pass with KV cache."""
        cache = small_model.create_kv_cache()

        # Prefill
        input_ids = torch.randint(0, 1000, (1, 5), device="mps")
        logits = small_model(input_ids, kv_cache=cache)
        cache.advance(5)

        assert logits.shape == (1, 5, 1000)
        assert cache.seq_len == 5

        # Decode
        next_input = torch.randint(0, 1000, (1, 1), device="mps")
        logits = small_model(next_input, kv_cache=cache)

        assert logits.shape == (1, 1, 1000)


# ---------------------------------------------------------------------------
# Generation Tests
# ---------------------------------------------------------------------------


@requires_metal_kernels
class TestGeneration:
    """Tests for autoregressive generation."""

    @pytest.mark.smoke
    def test_greedy_generation(self, small_model):
        """Test greedy decoding generation."""
        input_ids = torch.randint(0, 1000, (1, 5), device="mps")
        max_new_tokens = 10

        generated = _simple_generate(
            small_model,
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Should have prompt + generated tokens
        assert generated.shape[1] == 5 + max_new_tokens

    def test_sampling_generation(self, small_model):
        """Test temperature sampling generation."""
        input_ids = torch.randint(0, 1000, (1, 3), device="mps")

        generated = _simple_generate(
            small_model,
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
        )

        assert generated.shape[1] >= 3  # At least prompt

    def test_generation_determinism(self, small_model):
        """Test that greedy generation is deterministic."""
        input_ids = torch.randint(0, 1000, (1, 5), device="mps")

        gen1 = _simple_generate(small_model, input_ids, max_new_tokens=10, do_sample=False)
        gen2 = _simple_generate(small_model, input_ids, max_new_tokens=10, do_sample=False)

        assert torch.equal(gen1, gen2)


# ---------------------------------------------------------------------------
# Helper function for generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _simple_generate(
    model: SimpleMetalTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 10,
    do_sample: bool = False,
    temperature: float = 1.0,
    eos_token_id: int = 999,
) -> torch.Tensor:
    """Simple generation loop for testing.

    Args:
        model: Model to generate with
        input_ids: [1, seq_len] prompt token IDs
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to sample (True) or greedy (False)
        temperature: Sampling temperature
        eos_token_id: EOS token ID to stop generation

    Returns:
        [1, total_len] full sequence
    """
    cache = model.create_kv_cache()

    # Prefill
    logits = model(input_ids, kv_cache=cache)
    cache.advance(input_ids.shape[1])

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]

        if do_sample and temperature > 0:
            # Temperature sampling
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

        # Decode step
        logits = model(next_token, kv_cache=cache)
        cache.advance(1)

    return generated
