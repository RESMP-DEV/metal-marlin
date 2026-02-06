"""
High-level inference API for Marlin-quantized models.

This module provides a PyTorch-native pipeline for loading and running
FP4-quantized models on Apple Silicon via MPS. No MLX dependency.

Backend support:
    - torch: PyTorch MPS/CPU inference (required)
    - Falls back to CPU if MPS unavailable

Usage:
    from metal_marlin.inference import MarlinPipeline

    # Load pre-quantized model
    pipe = MarlinPipeline.from_pretrained("./glm4-fp4/")

    # Generate text
    output = pipe("What is the meaning of life?", max_tokens=100)
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from .._compat import HAS_MPS, require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing

    from ..kv_cache import KVCache as KVCacheTorch


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
        kv_cache: KVCacheTorch | None = None,
    ) -> torch_typing.Tensor:
        """Forward pass returning logits."""
        ...

    def create_kv_cache(self, batch_size: int = 1) -> KVCacheTorch:
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

    # Use CPU for dequantization on MPS to avoid unsupported ops and ensure
    # we can always move the result back to the requested device.
    target_device = packed.device
    work_device = target_device
    if target_device.type == "mps":
        packed = packed.to("cpu")
        scales = scales.to("cpu")
        work_device = packed.device

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
        device=work_device,
    )

    # Storage format from hf_loader.convert_model_to_fp4 is TRANSPOSED:
    # packed: [in_feat // 8, out_feat], scales: [n_groups, out_feat]
    # We need to transpose and unpack to get [out_feat, in_feat]

    packed_in = packed.shape[0]  # in_feat // 8
    out_feat = packed.shape[1]
    in_feat = packed_in * 8

    # Unpack 8 FP4 values from each uint32
    # packed is [in_feat // 8, out_feat], indices will be [in_feat, out_feat]
    indices = torch.zeros((in_feat, out_feat), dtype=torch.long, device=work_device)
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

    values = values.half()
    if values.device != target_device:
        values = values.to(target_device)
    return values


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
        self.group_size = int(config.get("group_size", 128))

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
            self._marlin_gemm_fp4 = marlin_gemm_fp4
        except Exception:
            self._use_fused_kernels = False
            self._marlin_gemm_fp4 = None

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
            packed_name = name
            packed_suffix = f"{name}.packed"
            if packed_name not in self.weights and packed_suffix in self.weights:
                packed_name = packed_suffix
            packed = self.weights[packed_name]
            scales = self.weights[scales_name]

            # Get group size
            if gs_name in self.weights:
                group_size = int(self.weights[gs_name].item())
            else:
                group_size = self.group_size

            # Dequantize
            weight = dequantize_fp4_torch(packed, scales, group_size)
            self._dequant_cache[name] = weight
            return weight

        # Not quantized, return as-is
        if name in self.weights:
            return self.weights[name]

        raise KeyError(f"Weight {name} not found")

    def _forward_linear(
        self,
        x: torch_typing.Tensor,
        layer_name: str,
    ) -> torch_typing.Tensor:
        """Linear layer with optional fused kernel."""
        assert torch is not None

        if self._use_fused_kernels and self.config.get("quant_type", "fp4") == "fp4":
            scales_name = f"{layer_name}.scales"
            gs_name = f"{layer_name}.group_size"

            if self._marlin_gemm_fp4 is not None and scales_name in self.weights:
                packed_name = layer_name
                if packed_name not in self.weights:
                    packed_name = f"{layer_name}.packed"

                if packed_name in self.weights:
                    group_size = self.group_size
                    if gs_name in self.weights:
                        group_size = int(self.weights[gs_name].item())

                    packed = self.weights[packed_name]
                    scales = self.weights[scales_name]
                    k_dim = x.shape[-1]
                    if (
                        packed.ndim == 2
                        and scales.ndim == 2
                        and k_dim % 8 == 0
                        and k_dim % group_size == 0
                        and packed.shape[0] == k_dim // 8
                        and scales.shape[0] == k_dim // group_size
                        and packed.shape[1] == scales.shape[1]
                    ):
                        return self._marlin_gemm_fp4(x, packed, scales, group_size)

        weight = self._get_weight(layer_name)
        return torch.nn.functional.linear(x, weight)

    def _quantized_linear(
        self,
        x: torch_typing.Tensor,
        weight_name: str,
    ) -> torch_typing.Tensor:
        """Backward-compatible alias for _forward_linear."""
        return self._forward_linear(x, weight_name)

    def __call__(
        self,
        input_ids: torch_typing.Tensor,
        kv_cache: KVCacheTorch | None = None,
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
        kv_cache: KVCacheTorch | None = None,
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

        # Try Metal implementation if available
        try:
            from ..layernorm_metal import require_metal, require_mps, rmsnorm_metal
            from ..metal_dispatch import HAS_METAL

            if HAS_METAL and x.is_mps and weight.is_mps:
                require_metal()
                require_mps()
                return rmsnorm_metal(x, weight, self.rms_norm_eps)
        except Exception:
            pass

        # Fallback to PyTorch implementation
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

        # Try Metal implementation if available
        try:
            from ..layernorm_metal import require_metal, require_mps, rmsnorm_metal
            from ..metal_dispatch import HAS_METAL

            if HAS_METAL and x.is_mps and weight.is_mps:
                require_metal()
                require_mps()
                # Reshape for per-head processing
                orig_shape = x.shape
                batch, heads, seq_len, head_dim = orig_shape
                x_reshaped = x.view(batch * heads * seq_len, head_dim)
                # Expand weight to match all tokens
                weight_expanded = weight.unsqueeze(0).expand(batch * heads * seq_len, -1)
                result = rmsnorm_metal(x_reshaped, weight_expanded, self.rms_norm_eps)
                return result.view(orig_shape)
        except Exception:
            pass

        # Fallback to PyTorch implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x * weight

    def _attention(
        self,
        hidden: torch_typing.Tensor,
        layer_idx: int,
        kv_cache: KVCacheTorch | None = None,
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

        # SwiGLU activation - use fused Metal kernel
        from ..activation_metal import _USE_METAL_ACTIVATION, swiglu_fused_metal

        if _USE_METAL_ACTIVATION and gate.is_mps:
            hidden = swiglu_fused_metal(gate, up)
        else:
            hidden = torch.nn.functional.silu(gate) * up

        return self._quantized_linear(hidden, f"{prefix}.down_proj.weight")

    def create_kv_cache(self, batch_size: int = 1) -> KVCacheTorch:
        """Create empty KV cache structure."""
        require_torch()

        from ..kv_cache import CacheConfig as CacheConfigTorch, KVCache as KVCacheTorch

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


def _load_new_checkpoint(
    path: str | Path,
    device: str,
    **kwargs: Any,
) -> tuple[Any, Any]:
    from .pipeline_v2 import TransformersMarlinPipeline

    pipeline = TransformersMarlinPipeline.from_pretrained(
        str(path),
        device=device,
        **kwargs,
    )
    return pipeline.model, pipeline.tokenizer


class MarlinPipeline:
    """
    DEPRECATED: Use TransformersMarlinPipeline instead.

    This class is maintained for backwards compatibility but internally
    now uses Transformers + layer replacement.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        device: str | None = "mps",
    ) -> None:
        warnings.warn(
            "MarlinPipeline is deprecated. Use TransformersMarlinPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from .pipeline_v2 import TransformersMarlinPipeline

        if isinstance(model, TransformersMarlinPipeline):
            self._pipeline = model
        else:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer is required when constructing MarlinPipeline from a model."
                )
            if device is None:
                device = getattr(model, "device", get_device())
            self._pipeline = TransformersMarlinPipeline(model, tokenizer, device=device)

        self.model = self._pipeline.model
        self.tokenizer = self._pipeline.tokenizer

    @property
    def device(self) -> str:
        """Return the active device for this pipeline."""
        return getattr(self._pipeline, "device", get_device())

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        quant_type: str = "fp4",
        device: str | None = None,
        **kwargs: Any,
    ) -> MarlinPipeline:
        """Load from quantized checkpoint directory or HF model id."""
        if device is None:
            device = get_device()
        if quant_type not in {"fp4", "int4"}:
            warnings.warn(
                f"Unsupported quant_type='{quant_type}' for Transformers-based pipeline; "
                "falling back to fp4 behavior.",
                UserWarning,
                stacklevel=2,
            )

        model, tokenizer = _load_new_checkpoint(path, device, **kwargs)
        return cls(model, tokenizer, device=device)

    def __call__(
        self,
        prompt: str | list[str],
        **kwargs: Any,
    ) -> str | Iterator[str] | list[str]:
        kwargs = dict(kwargs)
        stream = bool(kwargs.pop("stream", False))
        if isinstance(prompt, list):
            if stream:
                first_prompt = prompt[0] if prompt else ""
                return self._pipeline(first_prompt, stream=True, **kwargs)
            if hasattr(self._pipeline, "batch_generate"):
                return self._pipeline.batch_generate(prompt, **kwargs)
            first_prompt = prompt[0] if prompt else ""
            return self._pipeline(first_prompt, stream=False, **kwargs)
        return self._pipeline(prompt, stream=stream, **kwargs)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        """Yield tokens for streaming responses."""
        result = self(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True
        )
        return cast(Iterator[str], result)

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

        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}

        vocab_size = config_dict.get("vocab_size", getattr(config, "vocab_size", 32000))
        hidden_size = config_dict.get("hidden_size", getattr(config, "hidden_size", 4096))
        num_hidden_layers = config_dict.get(
            "num_hidden_layers",
            getattr(config, "num_hidden_layers", 32),
        )
        intermediate_size = config_dict.get(
            "intermediate_size",
            getattr(config, "intermediate_size", 11008),
        )
        quant_type = config_dict.get(
            "quant_type",
            getattr(config, "quant_type", "fp4"),
        )

        num_params = vocab_size * hidden_size + num_hidden_layers * (
            4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
        )

        bits_per_param = 4 if quant_type in ("fp4", "int4") else 16
        memory_mb = num_params * bits_per_param / 8 / 1024 / 1024

        return ModelInfo(
            name=f"Model-{num_params // 1e9:.0f}B",
            num_params=num_params,
            quant_type=quant_type,
            memory_mb=memory_mb,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)


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


__all__ = [
    "GenerationConfig",
    "MarlinModel",
    "MarlinPipeline",
    "MetalMarlinModel",
    "ModelConfig",
    "ModelInfo",
    "chat",
    "dequantize_fp4_torch",
    "get_device",
    "load_quantized_model",
    "load_safetensors_torch",
]
