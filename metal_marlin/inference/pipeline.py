"""
High-level inference API for Marlin-quantized models.

This module provides a PyTorch-native pipeline for loading and running
FP4-quantized models on Apple Silicon via MPS. No MLX dependency.

Backend support:
    - torch: PyTorch MPS/CPU inference (required)
    - Falls back to CPU if MPS unavailable

Usage:
    from metal_marlin.inference import MarlinPipeline, load_quantized_model

    # Load pre-quantized model
    model, tokenizer = load_quantized_model("./glm4-fp4/")

    # Generate text
    pipe = MarlinPipeline(model, tokenizer)
    output = pipe("What is the meaning of life?", max_tokens=100)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .._compat import HAS_MPS, require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


def get_device() -> str:
    """Return the best available device for inference.

    Priority:
        1. mps - Apple Silicon GPU
        2. cuda - NVIDIA GPU
        3. cpu - Fallback

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    require_torch()
    assert torch is not None

    if HAS_MPS:
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    eos_token_id: int = 2
    pad_token_id: int = 0
    do_sample: bool = True


@runtime_checkable
class MarlinModel(Protocol):
    """Protocol for models compatible with MarlinPipeline."""

    def __call__(
        self,
        input_ids: torch_typing.Tensor,
        kv_cache: Any = None,
    ) -> torch_typing.Tensor:
        """Forward pass returning logits."""
        ...

    def create_kv_cache(self, batch_size: int = 1) -> Any:
        """Create KV cache for incremental decoding."""
        ...


@dataclass
class ModelInfo:
    name: str
    num_params: int
    quant_type: str
    memory_mb: float


@dataclass
class ModelConfig:
    """Generic model configuration."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    quant_type: str = "fp4"
    hidden_act: str = "silu"
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    head_dim: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Create config from dictionary."""
        return cls(
            vocab_size=d.get("vocab_size", 32000),
            hidden_size=d.get("hidden_size", 4096),
            intermediate_size=d.get("intermediate_size", 11008),
            num_hidden_layers=d.get("num_hidden_layers", 32),
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", 32),
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 10000.0),
            quant_type=d.get("quant_type", "fp4"),
            hidden_act=d.get("hidden_act", "silu"),
            mlp_bias=d.get("mlp_bias", False),
            attention_bias=d.get("attention_bias", False),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            head_dim=d.get("head_dim"),
        )


def load_safetensors_torch(
    path: Path,
    device: str = "mps",
) -> dict[str, torch_typing.Tensor]:
    """Load safetensors file directly to PyTorch tensors.

    Args:
        path: Path to .safetensors file
        device: Target device for tensors

    Returns:
        Dict mapping tensor names to torch tensors on device
    """
    require_torch()
    assert torch is not None

    from safetensors.torch import load_file

    weights = load_file(str(path), device=device)
    return weights


def load_config(model_path: str | Path) -> dict[str, Any]:
    """Load config.json from a model directory."""
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def dequantize_fp4_torch(
    packed: torch_typing.Tensor,
    scales: torch_typing.Tensor,
    group_size: int,
) -> torch_typing.Tensor:
    """Dequantize FP4-packed weights to float16 using PyTorch.

    Args:
        packed: uint32 packed weights [out_feat, in_feat // 8]
        scales: float16 scales [out_feat, n_groups]
        group_size: Number of elements per group

    Returns:
        Dequantized float16 weights [out_feat, in_feat]
    """
    require_torch()
    assert torch is not None

    # FP4 E2M1 lookup table
    E2M1_VALUES = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )

    # Storage format from hf_loader.convert_model_to_fp4 is TRANSPOSED:
    # packed: [in_feat // 8, out_feat], scales: [n_groups, out_feat]
    # We need to transpose and unpack to get [out_feat, in_feat]

    packed_in = packed.shape[0]  # in_feat // 8
    out_feat = packed.shape[1]
    in_feat = packed_in * 8

    # Unpack 8 FP4 values from each uint32
    # packed is [in_feat // 8, out_feat], indices will be [in_feat, out_feat]
    indices = torch.zeros((in_feat, out_feat), dtype=torch.long, device=packed.device)
    # Convert to int64 for bitshift (uint32 rshift not implemented on CPU)
    packed_i64 = packed.to(torch.int64)
    for i in range(8):
        indices[i::8, :] = ((packed_i64 >> (i * 4)) & 0xF).long()

    # Dequantize using lookup table
    values = E2M1_VALUES[indices]  # [in_feat, out_feat]

    # Apply scales - scales is [n_groups, out_feat]
    n_groups = in_feat // group_size
    values = values.reshape(n_groups, group_size, out_feat)
    scales_expanded = scales[:, None, :].float()  # [n_groups, 1, out_feat]
    values = values * scales_expanded
    values = values.reshape(in_feat, out_feat)

    # Transpose to standard [out_feat, in_feat] layout
    values = values.T

    return values.half()


class MetalMarlinModel:
    """PyTorch-based model for Marlin FP4 quantized weights.

    Loads pre-quantized safetensors and runs inference on MPS.
    Supports autoregressive generation with KV cache.

    Attributes:
        config: Model configuration from config.json
        device: PyTorch device (mps, cuda, or cpu)

    Example:
        >>> model = MetalMarlinModel.from_quantized("./glm4-fp4/")
        >>> logits = model(input_ids)
    """

    def __init__(
        self,
        config: dict[str, Any],
        weights: dict[str, torch_typing.Tensor],
        device: str = "mps",
    ):
        require_torch()
        assert torch is not None

        self.config = config
        self.weights = weights
        self.device = device

        # Extract config values
        self.hidden_size = config.get("hidden_size", 4096)
        self.num_layers = config.get("num_hidden_layers", 32)
        self.num_heads = config.get("num_attention_heads", 32)
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.vocab_size = config.get("vocab_size", 32000)
        # Use explicit head_dim from config if available (e.g., Qwen3 has head_dim=128)
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)

        # RoPE (Rotary Position Embeddings) config
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.max_position_embeddings = config.get("max_position_embeddings", 4096)

        # Precompute RoPE frequencies
        self._rope_cache: dict[int, tuple[torch_typing.Tensor, torch_typing.Tensor]] = {}

        # Build dequantized weight cache (lazy)
        self._dequant_cache: dict[str, torch_typing.Tensor] = {}

        # Check if fused kernels are available and working
        try:
            from ..kernels import HAS_METAL, HAS_MPS, marlin_gemm_fp4

            self._use_fused_kernels = HAS_METAL and HAS_MPS
        except ImportError:
            self._use_fused_kernels = False

    @classmethod
    def from_quantized(
        cls,
        model_path: str | Path,
        device: str | None = None,
    ) -> MetalMarlinModel:
        """Load a pre-quantized model.

        Args:
            model_path: Path to directory containing model.safetensors and config.json
            device: Device to load to. If None, auto-detects best device.

        Returns:
            MetalMarlinModel instance ready for inference

        Raises:
            FileNotFoundError: If model files not found
        """
        require_torch()

        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Determine device
        if device is None:
            device = get_device()

        # Load weights via safetensors
        st_file = model_path / "model.safetensors"
        if not st_file.exists():
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")

        weights = load_safetensors_torch(st_file, device=device)

        return cls(config, weights, device=device)

    def _get_weight(self, name: str) -> torch_typing.Tensor:
        """Get a weight tensor, dequantizing if needed.

        Caches dequantized weights for reuse.
        """
        # Check cache first
        if name in self._dequant_cache:
            return self._dequant_cache[name]

        # Check if this is a quantized weight
        scales_name = f"{name}.scales"
        gs_name = f"{name}.group_size"

        if scales_name in self.weights:
            packed = self.weights[name]
            scales = self.weights[scales_name]

            # Get group size
            if gs_name in self.weights:
                group_size = int(self.weights[gs_name].item())
            else:
                group_size = 128

            # Dequantize
            weight = dequantize_fp4_torch(packed, scales, group_size)
            self._dequant_cache[name] = weight
            return weight

        # Not quantized, return as-is
        if name in self.weights:
            return self.weights[name]

        raise KeyError(f"Weight {name} not found")

    def _quantized_linear(
        self,
        x: torch_typing.Tensor,
        weight_name: str,
    ) -> torch_typing.Tensor:
        """Linear with quantized weights - uses fused kernel if available."""
        assert torch is not None

        if self._use_fused_kernels and self.config.get("quant_type", "fp4") == "fp4":
            packed_name = f"{weight_name}.packed"
            scales_name = f"{weight_name}.scales"
            gs_name = f"{weight_name}.group_size"

            if packed_name in self.weights and scales_name in self.weights:
                from ..kernels import marlin_gemm_fp4

                group_size = 128
                if gs_name in self.weights:
                    group_size = int(self.weights[gs_name].item())

                packed = self.weights[packed_name]
                scales = self.weights[scales_name]
                return marlin_gemm_fp4(x, packed, scales, group_size)

        weight = self._get_weight(weight_name)
        return torch.nn.functional.linear(x, weight)

    def __call__(
        self,
        input_ids: torch_typing.Tensor,
        kv_cache: Any = None,
    ) -> torch_typing.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            kv_cache: Optional KV cache for incremental decoding

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        require_torch()
        assert torch is not None

        # Ensure input is on correct device
        if input_ids.device.type != self.device:
            input_ids = input_ids.to(self.device)

        # Get embeddings
        embed_weight = self._get_weight("model.embed_tokens.weight")
        hidden = torch.nn.functional.embedding(input_ids, embed_weight)
        # Convert to half precision for efficient computation with quantized weights
        hidden = hidden.half()

        # Process through transformer layers
        for layer_idx in range(self.num_layers):
            hidden = self._forward_layer(hidden, layer_idx, kv_cache)

        # Final RMSNorm
        hidden = self._rms_norm(hidden, "model.norm.weight")

        # LM head projection (handle weight tying)
        if "lm_head.weight" in self.weights:
            lm_head_weight = self._get_weight("lm_head.weight")
        else:
            # Weight tying: use embed_tokens as lm_head
            lm_head_weight = self._get_weight("model.embed_tokens.weight")
        logits = torch.nn.functional.linear(hidden, lm_head_weight)

        return logits

    def _forward_layer(
        self,
        hidden: torch_typing.Tensor,
        layer_idx: int,
        kv_cache: Any = None,
    ) -> torch_typing.Tensor:
        """Forward pass through a single transformer layer."""
        assert torch is not None

        prefix = f"model.layers.{layer_idx}"

        # Pre-attention RMSNorm
        normed = self._rms_norm(hidden, f"{prefix}.input_layernorm.weight")

        # Self-attention
        attn_out = self._attention(normed, layer_idx, kv_cache)

        # Residual connection
        hidden = hidden + attn_out

        # Pre-MLP RMSNorm
        normed = self._rms_norm(hidden, f"{prefix}.post_attention_layernorm.weight")

        # MLP
        mlp_out = self._mlp(normed, layer_idx)

        # Residual connection
        hidden = hidden + mlp_out

        return hidden

    def _rms_norm(
        self,
        x: torch_typing.Tensor,
        weight_name: str,
    ) -> torch_typing.Tensor:
        """Apply RMSNorm."""
        assert torch is not None

        weight = self._get_weight(weight_name)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x * weight

    def _per_head_rms_norm(
        self,
        x: torch_typing.Tensor,
        weight: torch_typing.Tensor,
    ) -> torch_typing.Tensor:
        """Apply RMSNorm per attention head.

        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            weight: Norm weight [head_dim]
        """
        assert torch is not None

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x * weight

    def _attention(
        self,
        hidden: torch_typing.Tensor,
        layer_idx: int,
        kv_cache: Any = None,
    ) -> torch_typing.Tensor:
        """Compute self-attention with RoPE."""
        assert torch is not None

        batch_size, seq_len, _ = hidden.shape
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Q, K, V projections
        q = self._quantized_linear(hidden, f"{prefix}.q_proj.weight")
        k = self._quantized_linear(hidden, f"{prefix}.k_proj.weight")
        v = self._quantized_linear(hidden, f"{prefix}.v_proj.weight")

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Normalization (Qwen3 and similar models)
        if f"{prefix}.q_norm.weight" in self.weights:
            q_norm_weight = self._get_weight(f"{prefix}.q_norm.weight")
            k_norm_weight = self._get_weight(f"{prefix}.k_norm.weight")
            # Apply RMS norm per-head (weight is [head_dim])
            q = self._per_head_rms_norm(q, q_norm_weight)
            k = self._per_head_rms_norm(k, k_norm_weight)

        # Apply RoPE (Rotary Position Embeddings)
        # Get position offset from KV cache
        position_offset = kv_cache.seq_len if kv_cache is not None else 0
        cos, sin = self._get_rope_embeddings(seq_len, position_offset)
        q = self._apply_rotary_emb(q, cos, sin)
        k = self._apply_rotary_emb(k, cos, sin)

        # KV cache update (BEFORE GQA repeat - cache stores original num_kv_heads)
        if kv_cache is not None:
            k_full, v_full = kv_cache.update(layer_idx, k, v)
        else:
            k_full, v_full = k, v

        # Handle GQA: repeat K, V if num_kv_heads < num_heads (AFTER cache)
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(repeats, dim=1)
            v_full = v_full.repeat_interleave(repeats, dim=1)

        # Ensure consistent dtype for attention (fix potential float32/float16 mix)
        q = q.half()
        k_full = k_full.half()
        v_full = v_full.half()

        # Scaled dot-product attention
        # is_causal=True only works correctly when Q and K have the same seq_len.
        # During decode (single token query attending to cached K), we must use
        # is_causal=False because the single query should attend to ALL keys.
        is_decode = seq_len == 1 and kv_cache is not None
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            is_causal=not is_decode,
        )

        # Reshape back and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self._quantized_linear(attn_out, f"{prefix}.o_proj.weight")

    def _get_rope_embeddings(
        self, seq_len: int, position_offset: int = 0
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Get RoPE sin/cos embeddings for given sequence length and offset."""
        assert torch is not None

        total_len = seq_len + position_offset
        cache_key = total_len

        if cache_key not in self._rope_cache:
            # Compute inverse frequencies
            half_dim = self.head_dim // 2
            inv_freq = 1.0 / (
                self.rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
            )
            inv_freq = inv_freq.to(self.device)

            # Compute position embeddings
            positions = torch.arange(total_len, dtype=torch.float32, device=self.device)
            freqs = torch.outer(positions, inv_freq)  # [seq_len, half_dim]

            # Compute sin/cos
            cos = torch.cos(freqs).half()
            sin = torch.sin(freqs).half()

            self._rope_cache[cache_key] = (cos, sin)

        cos, sin = self._rope_cache[cache_key]
        return cos[position_offset:total_len], sin[position_offset:total_len]

    def _apply_rotary_emb(
        self, x: torch_typing.Tensor, cos: torch_typing.Tensor, sin: torch_typing.Tensor
    ) -> torch_typing.Tensor:
        """Apply rotary position embeddings to tensor x.

        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            cos: Cosine frequencies [seq_len, half_dim]
            sin: Sine frequencies [seq_len, half_dim]
        """
        assert torch is not None

        # Split x into first and second halves
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        # Broadcast cos/sin to match x shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        rotated = torch.cat(
            [
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ],
            dim=-1,
        )

        return rotated

    def _mlp(
        self,
        hidden: torch_typing.Tensor,
        layer_idx: int,
    ) -> torch_typing.Tensor:
        """Compute MLP (SwiGLU)."""
        assert torch is not None

        prefix = f"model.layers.{layer_idx}.mlp"

        gate = self._quantized_linear(hidden, f"{prefix}.gate_proj.weight")
        up = self._quantized_linear(hidden, f"{prefix}.up_proj.weight")

        # SwiGLU activation
        hidden = torch.nn.functional.silu(gate) * up

        return self._quantized_linear(hidden, f"{prefix}.down_proj.weight")

    def create_kv_cache(self, batch_size: int = 1) -> KVCache:
        """Create empty KV cache structure."""
        require_torch()

        from ..kv_cache_torch import CacheConfigTorch, KVCacheTorch

        config = CacheConfigTorch(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.config.get("max_position_embeddings", 4096),
            dtype="fp16",
        )
        return KVCacheTorch(config, batch_size=batch_size, device=self.device)

    def info(self) -> dict[str, Any]:
        """Return model information."""
        return {
            "model_type": self.config.get("model_type", "unknown"),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "device": self.device,
        }


# Type alias for backward compatibility
KVCache = Any


class MarlinPipeline:
    """
    High-level pipeline for text generation.

    Works with any model implementing the MarlinModel protocol.
    Uses PyTorch for tensor operations and MPS for GPU acceleration.

    Usage:
        model = MetalMarlinModel.from_quantized("./glm4-fp4/")
        pipe = MarlinPipeline(model, tokenizer)
        output = pipe("What is the meaning of life?", max_tokens=100)
    """

    def __init__(
        self,
        model: MarlinModel,
        tokenizer: Any,
        config: GenerationConfig | None = None,
        device: str | None = None,
    ):
        require_torch()

        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self._kv_cache = None

        # Get device from model or auto-detect
        if device is not None:
            self._device = device
        elif hasattr(model, "device"):
            self._device = model.device
        else:
            self._device = get_device()

    @property
    def device(self) -> str:
        """Return the active device for this pipeline."""
        return self._device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        quant_type: str = "fp4",
        device: str | None = None,
    ) -> MarlinPipeline:
        """
        Load a pretrained model from safetensors + config.json.

        Args:
            model_path: Path to model directory
            quant_type: Quantization type (fp4, int4) - informational only
            device: Device to load to (mps, cuda, cpu). Auto-detects if None.

        Returns:
            MarlinPipeline ready for inference
        """
        require_torch()

        model_path = Path(model_path)
        config = load_config(model_path)

        # Detect model architecture
        arch = config.get("architectures", [""])[0]

        if "GLM" in arch or config.get("model_type") == "glm4_moe_lite":
            from ..models.glm4_model import GLM4MoEModel

            model = GLM4MoEModel.from_quantized(model_path, device=device)
        elif "Qwen" in arch:
            model = MetalMarlinModel.from_quantized(model_path, device=device)
        else:
            model = MetalMarlinModel.from_quantized(model_path, device=device)

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return cls(model, tokenizer, device=model.device)

    def __call__(
        self,
        prompt: str | list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text or list of texts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stream: If True, return iterator yielding tokens

        Returns:
            Generated text or iterator
        """
        require_torch()
        assert torch is not None

        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self._device)

        # Update config
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id or 2,
        )

        if stream:
            return self._stream_generate(input_ids, gen_config)
        else:
            return self._generate(input_ids, gen_config)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        """Yield tokens as they are generated for streaming responses."""
        require_torch()
        assert torch is not None

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        kv_cache = self.model.create_kv_cache(batch_size=1)

        # Prefill
        logits = self.model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])

        eos_token_id = self.tokenizer.eos_token_id or 2

        for _ in range(max_tokens):
            next_token_id = self._sample(
                logits[0, -1],
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            token_str = self.tokenizer.decode([next_token_id])
            yield token_str

            if next_token_id == eos_token_id:
                break

            # Decode step
            next_input = torch.tensor([[next_token_id]], device=self._device, dtype=torch.long)
            logits = self.model(next_input, kv_cache=kv_cache)
            kv_cache.advance(1)

    def _generate(
        self,
        input_ids: torch_typing.Tensor,
        config: GenerationConfig,
    ) -> str:
        """Standard autoregressive generation."""
        require_torch()
        assert torch is not None

        # Create KV cache for efficient decoding
        kv_cache = self.model.create_kv_cache(batch_size=1)

        # Prefill: process entire prompt
        logits = self.model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])

        generated_ids = input_ids.tolist()[0]

        for _ in range(config.max_new_tokens):
            next_token_id = self._sample(
                logits[:, -1, :][0],
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
            )
            generated_ids.append(next_token_id)

            # Check for EOS
            if next_token_id == config.eos_token_id:
                break

            # Decode step
            next_input = torch.tensor([[next_token_id]], device=self._device, dtype=torch.long)
            logits = self.model(next_input, kv_cache=kv_cache)
            kv_cache.advance(1)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _stream_generate(
        self,
        input_ids: torch_typing.Tensor,
        config: GenerationConfig,
    ) -> Iterator[str]:
        """Streaming generation yielding tokens one at a time."""
        require_torch()
        assert torch is not None

        kv_cache = self.model.create_kv_cache(batch_size=1)

        # Prefill
        logits = self.model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(input_ids.shape[1])

        generated_ids = input_ids.tolist()[0]
        prev_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        for _ in range(config.max_new_tokens):
            next_token_id = self._sample(
                logits[:, -1, :][0],
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
            )

            if next_token_id == config.eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Yield new text
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            yield full_text[len(prev_text) :]
            prev_text = full_text

            # Decode step
            next_input = torch.tensor([[next_token_id]], device=self._device, dtype=torch.long)
            logits = self.model(next_input, kv_cache=kv_cache)
            kv_cache.advance(1)

    def _sample(
        self,
        logits: torch_typing.Tensor,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> int:
        """Sample a token id from logits with temperature and top-p."""
        assert torch is not None

        if temperature > 0:
            logits = logits / temperature

        if do_sample and temperature > 0:
            probs = torch.nn.functional.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
            else:
                next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        return int(next_token.item())

    def info(self) -> ModelInfo:
        """Get model information."""
        config = getattr(self.model, "config", None)
        if config is None:
            return ModelInfo(
                name="Unknown",
                num_params=0,
                quant_type="unknown",
                memory_mb=0.0,
            )

        if isinstance(config, dict):
            vocab_size = config.get("vocab_size", 32000)
            hidden_size = config.get("hidden_size", 4096)
            num_hidden_layers = config.get("num_hidden_layers", 32)
            intermediate_size = config.get("intermediate_size", 11008)
            quant_type = config.get("quant_type", "fp4")
        else:
            vocab_size = getattr(config, "vocab_size", 32000)
            hidden_size = getattr(config, "hidden_size", 4096)
            num_hidden_layers = getattr(config, "num_hidden_layers", 32)
            intermediate_size = getattr(config, "intermediate_size", 11008)
            quant_type = getattr(config, "quant_type", "fp4")

        num_params = (
            vocab_size * hidden_size  # Embeddings
            + num_hidden_layers
            * (
                4 * hidden_size * hidden_size  # Attention
                + 3 * hidden_size * intermediate_size  # MLP
            )
        )

        if quant_type in ("fp4", "int4"):
            bits_per_param = 4
        else:
            bits_per_param = 16

        memory_mb = num_params * bits_per_param / 8 / 1024 / 1024

        return ModelInfo(
            name=f"Model-{num_params // 1e9:.0f}B",
            num_params=num_params,
            quant_type=quant_type,
            memory_mb=memory_mb,
        )


def load_quantized_model(
    model_path: str | Path,
    device: str | None = None,
) -> tuple[MetalMarlinModel, Any]:
    """Load a pre-quantized model and tokenizer.

    Convenience function that returns both model and tokenizer.

    Args:
        model_path: Path to model directory
        device: Device to load to (mps, cuda, cpu)

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoTokenizer

    model = MetalMarlinModel.from_quantized(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def chat(
    model_path: str,
    system_prompt: str = "You are a helpful assistant.",
) -> None:
    """Interactive chat interface."""
    require_torch()

    pipe = MarlinPipeline.from_pretrained(model_path)
    print(f"Loaded: {pipe.info()}")
    print(f"Device: {pipe.device}")
    print("Type 'quit' to exit.\n")

    history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # Build prompt with history
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)

        # Generate
        print("Assistant: ", end="", flush=True)
        response = ""
        for token in pipe(prompt, stream=True):
            print(token, end="", flush=True)
            response += token
        print()

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


# Backward compatibility aliases
MetalGLM47Model = MetalMarlinModel


__all__ = [
    "GenerationConfig",
    "MarlinModel",
    "MarlinPipeline",
    "MetalMarlinModel",
    "MetalGLM47Model",
    "ModelConfig",
    "ModelInfo",
    "chat",
    "dequantize_fp4_torch",
    "get_device",
    "load_quantized_model",
    "load_safetensors_torch",
]
