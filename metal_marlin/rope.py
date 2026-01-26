"""YaRN RoPE (Rotary Position Embedding) implementation.

YaRN (Yet another RoPE extensioN) extends the context length of transformers by
applying NTK-aware interpolation with attention temperature scaling.

Key features:
- NTK-aware interpolation: Different frequency bands scale differently
- mscale: Attention temperature compensation for longer contexts
- beta_fast/beta_slow: Control the interpolation ramp between low and high frequencies

Reference:
    "YaRN: Efficient Context Window Extension of Large Language Models"
    https://arxiv.org/abs/2309.00071

Usage:
    from metal_marlin.rope import YaRNConfig, compute_yarn_inv_freq, get_yarn_mscale

    # Parse from HF config
    config = YaRNConfig.from_hf_config(hf_config_dict)

    # Compute scaled inverse frequencies
    inv_freq = compute_yarn_inv_freq(
        dim=128,
        base=10000.0,
        config=config,
        device="mps",
    )

    # Get attention temperature scale
    mscale = get_yarn_mscale(config.scale_factor, config.mscale_all_dim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._compat import require_torch, torch


@dataclass
class YaRNConfig:
    """Configuration for YaRN RoPE scaling.

    Attributes:
        original_max_position: Original max position embeddings the model was trained with.
        scale_factor: Context extension factor (e.g., 4.0 = 4x longer context).
        beta_fast: Upper bound frequency for interpolation ramp. Default: 32.0.
            Frequencies above this threshold use linear interpolation.
        beta_slow: Lower bound frequency for interpolation ramp. Default: 1.0.
            Frequencies below this threshold are not interpolated (NTK-aware).
        mscale: Base attention temperature multiplier. Default: 1.0.
        mscale_all_dim: Scaling coefficient for mscale computation. Default: 0.0.
            When > 0, mscale = 0.1 * mscale_all_dim * log(scale_factor) + 1.0
            When 0, mscale is computed from scale_factor directly.
        attention_factor: Optional explicit attention scaling factor.
            If set, overrides mscale computation.
        rope_type: Type of RoPE scaling ("yarn", "linear", "dynamic", "ntk", "none").
    """

    original_max_position: int
    scale_factor: float
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    attention_factor: float | None = None
    rope_type: str = "yarn"

    def __post_init__(self) -> None:
        if self.scale_factor < 1.0:
            raise ValueError(f"scale_factor must be >= 1.0, got {self.scale_factor}")
        if self.beta_fast <= self.beta_slow:
            raise ValueError(
                f"beta_fast ({self.beta_fast}) must be > beta_slow ({self.beta_slow})"
            )

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> YaRNConfig | None:
        """Parse YaRN config from HuggingFace config.json.

        Handles various config formats:
        - rope_scaling dict with type="yarn"
        - Top-level rope_scaling_* fields
        - Model-specific formats (Qwen2, DeepSeek, etc.)

        Args:
            config: Dictionary loaded from config.json.

        Returns:
            YaRNConfig if rope_scaling is enabled, None otherwise.
        """
        rope_scaling = config.get("rope_scaling")

        # No rope_scaling - check for legacy fields
        if rope_scaling is None:
            # Check for top-level fields (some converters flatten the config)
            if config.get("rope_scaling_type") == "yarn":
                return cls(
                    original_max_position=config.get(
                        "rope_original_max_position",
                        config.get("original_max_position_embeddings", 4096),
                    ),
                    scale_factor=config.get("rope_scaling_factor", 1.0),
                    beta_fast=config.get("rope_beta_fast", 32.0),
                    beta_slow=config.get("rope_beta_slow", 1.0),
                    mscale=config.get("rope_mscale", 1.0),
                    mscale_all_dim=config.get("rope_mscale_all_dim", 0.0),
                    rope_type="yarn",
                )
            return None

        # rope_scaling is a dict
        if not isinstance(rope_scaling, dict):
            return None

        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type", "none"))
        if rope_type not in ("yarn", "longrope", "llama3"):
            # Not YaRN scaling
            return None

        # Parse rope_scaling dict fields
        factor = rope_scaling.get("factor", rope_scaling.get("scale_factor", 1.0))
        original_max = rope_scaling.get(
            "original_max_position_embeddings",
            config.get("original_max_position_embeddings", 4096),
        )

        # YaRN-specific parameters
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        mscale = rope_scaling.get("mscale", 1.0)
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
        attention_factor = rope_scaling.get("attention_factor")

        return cls(
            original_max_position=int(original_max),
            scale_factor=float(factor),
            beta_fast=float(beta_fast),
            beta_slow=float(beta_slow),
            mscale=float(mscale),
            mscale_all_dim=float(mscale_all_dim),
            attention_factor=float(attention_factor) if attention_factor is not None else None,
            rope_type=rope_type,
        )


def get_yarn_mscale(scale_factor: float, mscale_all_dim: float = 0.0) -> float:
    """Compute YaRN attention temperature scaling factor.

    The mscale compensates for the reduced attention entropy at longer contexts
    by scaling the softmax temperature. This helps maintain attention sharpness.

    Formula:
        If mscale_all_dim > 0:
            mscale = 0.1 * mscale_all_dim * log(scale_factor) + 1.0
        Else:
            mscale = 0.1 * log(scale_factor) + 1.0

    Args:
        scale_factor: Context extension factor (e.g., 4.0 for 4x context).
        mscale_all_dim: Scaling coefficient. Default: 0.0.

    Returns:
        Attention temperature multiplier (>= 1.0 for scale_factor >= 1.0).
    """
    if scale_factor <= 1.0:
        return 1.0

    if mscale_all_dim > 0:
        return 0.1 * mscale_all_dim * math.log(scale_factor) + 1.0
    else:
        return 0.1 * math.log(scale_factor) + 1.0


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 4096,
) -> float:
    """Find the RoPE dimension corresponding to a given number of rotations.

    Used to determine which dimension indices fall within the interpolation range
    defined by beta_fast and beta_slow.

    Args:
        num_rotations: Target number of rotations.
        dim: Total RoPE dimension.
        base: RoPE base frequency.
        max_position_embeddings: Original max position embeddings.

    Returns:
        Dimension index (float) corresponding to the target rotations.
    """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 4096,
) -> tuple[int, int]:
    """Find the dimension range for YaRN interpolation.

    Dimensions below low_idx are not interpolated (high-frequency, NTK-aware).
    Dimensions above high_idx use full linear interpolation (low-frequency).
    Dimensions in between use a smooth ramp.

    Args:
        beta_fast: Upper frequency bound (rotations).
        beta_slow: Lower frequency bound (rotations).
        dim: Total RoPE dimension.
        base: RoPE base frequency.
        max_position_embeddings: Original max position embeddings.

    Returns:
        (low_idx, high_idx) tuple defining the interpolation range.
    """
    low = math.floor(
        _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: int,
    high: int,
    dim: int,
    dtype: Any = None,
) -> Any:
    """Create a linear ramp mask for smooth interpolation blending.

    Values ramp linearly from 0 at low to 1 at high.
    Indices < low get 0 (no interpolation).
    Indices > high get 1 (full interpolation).

    Args:
        low: Start of ramp (value = 0).
        high: End of ramp (value = 1).
        dim: Total dimension.
        dtype: Output dtype (torch dtype).

    Returns:
        Tensor of shape [dim // 2] with ramp values.
    """
    require_torch("YaRN RoPE")
    if dtype is None:
        dtype = torch.float32

    if low == high:
        high = low + 0.001  # Avoid division by zero

    linear_func = (torch.arange(dim // 2, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0.0, 1.0)


def compute_yarn_inv_freq(
    dim: int,
    base: float,
    config: YaRNConfig,
    device: str = "mps",
) -> Any:
    """Compute YaRN-scaled inverse frequencies for RoPE.

    Implements NTK-aware interpolation where different frequency bands are
    scaled differently:
    - High frequencies (fast rotations): Not interpolated, preserve local patterns
    - Low frequencies (slow rotations): Linearly interpolated for extended context
    - Mid frequencies: Smooth blend via linear ramp

    This is more effective than uniform linear interpolation because high-frequency
    components encode local positional information that doesn't need scaling.

    Args:
        dim: RoPE embedding dimension (typically head_dim).
        base: RoPE base frequency (e.g., 10000.0).
        config: YaRN configuration with scale factor and beta parameters.
        device: Target device ("mps", "cuda", "cpu").

    Returns:
        Tensor of shape [dim // 2] containing scaled inverse frequencies.

    Example:
        >>> config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        >>> inv_freq = compute_yarn_inv_freq(128, 10000.0, config, device="mps")
        >>> # Shape: [64] - one inverse frequency per pair of dimensions
    """
    require_torch("YaRN RoPE")

    # Determine device
    if device == "mps" and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    # Base inverse frequencies (standard RoPE)
    # inv_freq[i] = 1 / (base^(2i/dim)) for i in [0, dim//2)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch_device) / dim)
    inv_freq = 1.0 / pos_freqs

    # Find the correction range based on beta_fast/beta_slow
    low, high = _yarn_find_correction_range(
        config.beta_fast,
        config.beta_slow,
        dim,
        base,
        config.original_max_position,
    )

    # Create linear ramp mask
    ramp_mask = _yarn_linear_ramp_mask(low, high, dim, dtype=torch.float32).to(torch_device)

    # NTK-aware scaling:
    # - extrapolation_factor = 1 (high freq, no scaling)
    # - interpolation_factor = 1/scale_factor (low freq, full interpolation)
    # - Blend using ramp_mask
    extrapolation_factor = 1.0
    interpolation_factor = 1.0 / config.scale_factor

    # Blend between extrapolation and interpolation
    # ramp_mask = 0: use extrapolation (high freq, preserve)
    # ramp_mask = 1: use interpolation (low freq, scale down)
    inv_freq_interpolated = (
        ramp_mask * interpolation_factor * inv_freq
        + (1 - ramp_mask) * extrapolation_factor * inv_freq
    )

    return inv_freq_interpolated


def compute_yarn_cos_sin_cache(
    dim: int,
    max_seq_len: int,
    base: float,
    config: YaRNConfig,
    device: str = "mps",
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Precompute cos/sin cache for YaRN RoPE embeddings.

    Caches cos(m * theta) and sin(m * theta) for all positions m up to max_seq_len,
    where theta is the YaRN-scaled inverse frequency.

    Args:
        dim: RoPE embedding dimension.
        max_seq_len: Maximum sequence length to cache.
        base: RoPE base frequency.
        config: YaRN configuration.
        device: Target device.
        dtype: Output dtype. Defaults to float32.

    Returns:
        (cos_cache, sin_cache) tensors of shape [max_seq_len, dim // 2].
    """
    require_torch("YaRN RoPE cache")

    if dtype is None:
        dtype = torch.float32

    # Compute scaled inverse frequencies
    inv_freq = compute_yarn_inv_freq(dim, base, config, device)

    # Determine device
    if device == "mps" and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    # Position indices
    t = torch.arange(max_seq_len, dtype=torch.float32, device=torch_device)

    # Compute angles: freqs[i, j] = t[i] * inv_freq[j]
    freqs = torch.outer(t, inv_freq)

    # Apply mscale to frequencies if applicable
    mscale = get_yarn_mscale(config.scale_factor, config.mscale_all_dim)
    if config.attention_factor is not None:
        mscale = config.attention_factor

    # Compute cos/sin with mscale applied
    cos_cache = torch.cos(freqs).to(dtype) * mscale
    sin_cache = torch.sin(freqs).to(dtype) * mscale

    return cos_cache, sin_cache


class YaRNRoPE:
    """YaRN RoPE embeddings with cached cos/sin values.

    This class provides efficient RoPE application with YaRN scaling.
    Cos/sin values are precomputed and cached for fast lookup during inference.

    Usage:
        rope = YaRNRoPE(dim=128, max_seq_len=8192, base=10000.0, config=yarn_config)
        q_rotated = rope.apply(q, offset=0)
        k_rotated = rope.apply(k, offset=kv_cache_len)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        config: YaRNConfig | None = None,
        device: str = "mps",
        dtype: Any = None,
    ):
        """Initialize YaRN RoPE.

        Args:
            dim: RoPE dimension (typically head_dim).
            max_seq_len: Maximum sequence length to support.
            base: RoPE base frequency.
            config: YaRN configuration. If None, uses standard RoPE (scale_factor=1).
            device: Target device.
            dtype: Cache dtype.
        """
        require_torch("YaRNRoPE")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Default config for standard RoPE
        if config is None:
            config = YaRNConfig(
                original_max_position=max_seq_len,
                scale_factor=1.0,
                beta_fast=32.0,
                beta_slow=1.0,
            )
        self.config = config

        # Compute and cache cos/sin
        self._build_cache()

    def _build_cache(self) -> None:
        """Build the cos/sin cache."""
        self.cos_cache, self.sin_cache = compute_yarn_cos_sin_cache(
            self.dim,
            self.max_seq_len,
            self.base,
            self.config,
            self.device,
            self.dtype,
        )

    def apply(
        self,
        x: Any,
        offset: int = 0,
    ) -> Any:
        """Apply YaRN RoPE to input tensor.

        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim].
            offset: Position offset for KV cache (used during autoregressive decoding).

        Returns:
            Tensor with RoPE applied, same shape as input.
        """
        require_torch("YaRNRoPE.apply")

        seq_len = x.shape[2]
        head_dim = x.shape[3]

        # Get cached cos/sin for the relevant positions
        cos = self.cos_cache[offset : offset + seq_len]  # [seq_len, dim//2]
        sin = self.sin_cache[offset : offset + seq_len]

        # Reshape for broadcasting: [1, 1, seq_len, dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split x into even and odd indices
        x_even = x[..., ::2]  # [batch, heads, seq, dim//2]
        x_odd = x[..., 1::2]

        # Apply rotation (Llama-style interleaved)
        # x_rotated = x * cos + rotate_half(x) * sin
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.view(*x.shape[:-1], head_dim)

        return x_rotated.to(x.dtype)

    def extend_cache(self, new_max_seq_len: int) -> None:
        """Extend the cache to support longer sequences.

        Args:
            new_max_seq_len: New maximum sequence length.
        """
        if new_max_seq_len <= self.max_seq_len:
            return

        self.max_seq_len = new_max_seq_len
        self._build_cache()


def create_rope_from_config(
    config: dict[str, Any],
    device: str = "mps",
) -> YaRNRoPE | None:
    """Create YaRN RoPE instance from HuggingFace config.

    Convenience function that parses the config and creates an appropriate
    RoPE instance. Returns None if no RoPE scaling is configured.

    Args:
        config: HuggingFace config.json dictionary.
        device: Target device.

    Returns:
        YaRNRoPE instance, or None if no scaling configured.
    """
    yarn_config = YaRNConfig.from_hf_config(config)
    if yarn_config is None:
        return None

    # Extract dimensions from config
    hidden_size = config.get("hidden_size", config.get("d_model", 4096))
    num_heads = config.get("num_attention_heads", config.get("n_head", 32))
    head_dim = config.get("head_dim", hidden_size // num_heads)
    max_position = config.get("max_position_embeddings", 8192)
    rope_theta = config.get("rope_theta", 10000.0)

    return YaRNRoPE(
        dim=head_dim,
        max_seq_len=max_position,
        base=rope_theta,
        config=yarn_config,
        device=device,
    )


# ---------------------------------------------------------------------------
# Metal kernel dispatch for YaRN RoPE
# ---------------------------------------------------------------------------

# Check for Metal dispatch availability
try:
    from .metal_dispatch import (
        HAS_METAL,
        HAS_MPS,
        MetalKernelLibrary,
        dispatch_kernel,
        get_default_library,
        mps_tensor_to_metal_buffer,
    )

    HAS_METAL_DISPATCH = HAS_METAL and HAS_MPS
except ImportError:
    HAS_METAL_DISPATCH = False
    HAS_METAL = False
    HAS_MPS = False


def dispatch_rope_yarn_forward(
    lib: Any,
    x: Any,
    cos_cache: Any,
    sin_cache: Any,
    attention_scale: float,
    position_offset: int = 0,
) -> Any:
    """Dispatch YaRN RoPE forward kernel on Metal.

    Args:
        lib: MetalKernelLibrary instance
        x: Input tensor [batch, seq_len, num_heads, head_dim], MPS tensor
        cos_cache: Precomputed YaRN cos cache [max_seq, dim/2], MPS tensor
        sin_cache: Precomputed YaRN sin cache [max_seq, dim/2], MPS tensor
        attention_scale: YaRN attention scaling factor (mscale)
        position_offset: Position offset for KV cache continuation

    Returns:
        Output tensor with YaRN RoPE applied [batch, seq_len, num_heads, head_dim]
    """
    require_torch("dispatch_rope_yarn_forward")

    if not HAS_METAL_DISPATCH:
        raise RuntimeError("Metal dispatch not available")

    import Metal
    import numpy as np

    device = lib.device

    batch_size, seq_len, num_heads, head_dim = x.shape
    half_head_dim = head_dim // 2

    # Ensure fp16 and contiguous
    x_fp16 = x.half().contiguous()
    cos_fp16 = cos_cache.half().contiguous()
    sin_fp16 = sin_cache.half().contiguous()

    # Allocate output
    output = torch.empty_like(x_fp16)

    # Convert to Metal buffers
    input_buf = mps_tensor_to_metal_buffer(x_fp16, device)
    cos_buf = mps_tensor_to_metal_buffer(cos_fp16, device)
    sin_buf = mps_tensor_to_metal_buffer(sin_fp16, device)
    output_buf = mps_tensor_to_metal_buffer(output, device)

    # Create constant buffers
    batch_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    seq_buf = device.newBufferWithBytes_length_options_(
        np.array([seq_len], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    heads_buf = device.newBufferWithBytes_length_options_(
        np.array([num_heads], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    dim_buf = device.newBufferWithBytes_length_options_(
        np.array([head_dim], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    offset_buf = device.newBufferWithBytes_length_options_(
        np.array([position_offset], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    scale_buf = device.newBufferWithBytes_length_options_(
        np.array([attention_scale], dtype=np.float32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Dispatch kernel
    grid = (half_head_dim, num_heads, batch_size * seq_len)
    threadgroup = (min(half_head_dim, 32), 1, 1)

    dispatch_kernel(
        lib,
        function_name="rope_yarn_forward",
        grid=grid,
        threadgroup=threadgroup,
        buffers=[
            input_buf,
            cos_buf,
            sin_buf,
            output_buf,
            batch_buf,
            seq_buf,
            heads_buf,
            dim_buf,
            offset_buf,
            scale_buf,
        ],
        wait=True,
    )

    return output


def dispatch_rope_yarn_latent(
    lib: Any,
    x: Any,
    cos_cache: Any,
    sin_cache: Any,
    attention_scale: float,
    kv_lora_rank: int,
    rope_dim: int,
    position_offset: int = 0,
) -> Any:
    """Dispatch YaRN RoPE for MLA latent representations.

    This kernel applies YaRN RoPE only to the positional portion of the
    combined latent representation (kv_a_proj output), leaving the
    kv_lora_rank portion unchanged.

    Args:
        lib: MetalKernelLibrary instance
        x: Input tensor [batch, seq_len, kv_lora_rank + rope_dim], MPS tensor
        cos_cache: Precomputed YaRN cos cache [max_seq, rope_dim/2], MPS tensor
        sin_cache: Precomputed YaRN sin cache [max_seq, rope_dim/2], MPS tensor
        attention_scale: YaRN attention scaling factor
        kv_lora_rank: Dimension of latent portion (passes through unchanged)
        rope_dim: Dimension of positional portion (gets RoPE applied)
        position_offset: Position offset for KV cache continuation

    Returns:
        Output tensor with YaRN RoPE applied to positional portion
    """
    require_torch("dispatch_rope_yarn_latent")

    if not HAS_METAL_DISPATCH:
        raise RuntimeError("Metal dispatch not available")

    import Metal
    import numpy as np

    device = lib.device

    batch_size, seq_len, total_dim = x.shape
    if total_dim != kv_lora_rank + rope_dim:
        raise ValueError(
            f"Input dim {total_dim} != kv_lora_rank {kv_lora_rank} + rope_dim {rope_dim}"
        )

    # Ensure fp16 and contiguous
    x_fp16 = x.half().contiguous()
    cos_fp16 = cos_cache.half().contiguous()
    sin_fp16 = sin_cache.half().contiguous()

    # Allocate output
    output = torch.empty_like(x_fp16)

    # Convert to Metal buffers
    input_buf = mps_tensor_to_metal_buffer(x_fp16, device)
    cos_buf = mps_tensor_to_metal_buffer(cos_fp16, device)
    sin_buf = mps_tensor_to_metal_buffer(sin_fp16, device)
    output_buf = mps_tensor_to_metal_buffer(output, device)

    # Create constant buffers
    batch_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    seq_buf = device.newBufferWithBytes_length_options_(
        np.array([seq_len], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    lora_buf = device.newBufferWithBytes_length_options_(
        np.array([kv_lora_rank], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    rope_buf = device.newBufferWithBytes_length_options_(
        np.array([rope_dim], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    offset_buf = device.newBufferWithBytes_length_options_(
        np.array([position_offset], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    scale_buf = device.newBufferWithBytes_length_options_(
        np.array([attention_scale], dtype=np.float32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Choose kernel based on rope_dim size
    half_rope_dim = rope_dim // 2
    if rope_dim <= 64:
        # Use simdgroup-optimized kernel for small rope_dim
        grid = (batch_size * seq_len, 1, 1)
        threadgroup = (32, 1, 1)  # One simdgroup
        kernel_name = "rope_yarn_latent_small"
    else:
        # Use general kernel for larger rope_dim
        grid = (total_dim, batch_size * seq_len, 1)
        threadgroup = (min(total_dim, 128), 1, 1)
        kernel_name = "rope_yarn_latent"

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=grid,
        threadgroup=threadgroup,
        buffers=[
            input_buf,
            cos_buf,
            sin_buf,
            output_buf,
            batch_buf,
            seq_buf,
            lora_buf,
            rope_buf,
            offset_buf,
            scale_buf,
        ],
        wait=True,
    )

    return output


def dispatch_rope_yarn_qk_fused(
    lib: Any,
    q: Any,
    k: Any,
    cos_cache: Any,
    sin_cache: Any,
    attention_scale: float,
    position_offset: int = 0,
) -> tuple[Any, Any]:
    """Dispatch fused YaRN RoPE for Q and K tensors.

    Applies YaRN RoPE to both Q and K in a single kernel dispatch,
    reducing dispatch overhead for attention computation.

    Args:
        lib: MetalKernelLibrary instance
        q: Query tensor [batch, seq_len, num_heads, head_dim], MPS tensor
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim], MPS tensor
        cos_cache: Precomputed YaRN cos cache [max_seq, dim/2], MPS tensor
        sin_cache: Precomputed YaRN sin cache [max_seq, dim/2], MPS tensor
        attention_scale: YaRN attention scaling factor
        position_offset: Position offset for KV cache continuation

    Returns:
        Tuple of (q_rotated, k_rotated) tensors
    """
    require_torch("dispatch_rope_yarn_qk_fused")

    if not HAS_METAL_DISPATCH:
        raise RuntimeError("Metal dispatch not available")

    import Metal
    import numpy as np

    device = lib.device

    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    half_head_dim = head_dim // 2

    # Ensure fp16 and contiguous
    q_fp16 = q.half().contiguous()
    k_fp16 = k.half().contiguous()
    cos_fp16 = cos_cache.half().contiguous()
    sin_fp16 = sin_cache.half().contiguous()

    # Allocate outputs
    q_output = torch.empty_like(q_fp16)
    k_output = torch.empty_like(k_fp16)

    # Convert to Metal buffers
    q_buf = mps_tensor_to_metal_buffer(q_fp16, device)
    k_buf = mps_tensor_to_metal_buffer(k_fp16, device)
    cos_buf = mps_tensor_to_metal_buffer(cos_fp16, device)
    sin_buf = mps_tensor_to_metal_buffer(sin_fp16, device)
    q_out_buf = mps_tensor_to_metal_buffer(q_output, device)
    k_out_buf = mps_tensor_to_metal_buffer(k_output, device)

    # Create constant buffers
    batch_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_size], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    seq_buf = device.newBufferWithBytes_length_options_(
        np.array([seq_len], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    heads_buf = device.newBufferWithBytes_length_options_(
        np.array([num_heads], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    kv_heads_buf = device.newBufferWithBytes_length_options_(
        np.array([num_kv_heads], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    dim_buf = device.newBufferWithBytes_length_options_(
        np.array([head_dim], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    offset_buf = device.newBufferWithBytes_length_options_(
        np.array([position_offset], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    scale_buf = device.newBufferWithBytes_length_options_(
        np.array([attention_scale], dtype=np.float32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Dispatch kernel
    max_heads = max(num_heads, num_kv_heads)
    grid = (half_head_dim, max_heads, batch_size * seq_len)
    threadgroup = (min(half_head_dim, 32), 1, 1)

    dispatch_kernel(
        lib,
        function_name="rope_yarn_qk_fused",
        grid=grid,
        threadgroup=threadgroup,
        buffers=[
            q_buf,
            k_buf,
            cos_buf,
            sin_buf,
            q_out_buf,
            k_out_buf,
            batch_buf,
            seq_buf,
            heads_buf,
            kv_heads_buf,
            dim_buf,
            offset_buf,
            scale_buf,
        ],
        wait=True,
    )

    return q_output, k_output


class YaRNRoPEMetal(YaRNRoPE):
    """YaRN RoPE with Metal kernel acceleration.

    Extends YaRNRoPE to use Metal shaders for RoPE computation when available.
    Falls back to PyTorch implementation when Metal is not available.

    Usage:
        rope = YaRNRoPEMetal(dim=128, max_seq_len=131072, config=yarn_config)
        q_rotated = rope.apply(q, offset=0)  # Uses Metal if available
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        config: YaRNConfig | None = None,
        device: str = "mps",
        dtype: Any = None,
    ):
        super().__init__(dim, max_seq_len, base, config, device, dtype)

        self._lib: Any | None = None
        if HAS_METAL_DISPATCH:
            try:
                self._lib = get_default_library()
            except Exception:
                pass

    def apply(
        self,
        x: Any,
        offset: int = 0,
    ) -> Any:
        """Apply YaRN RoPE using Metal kernel if available.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim] or
               [batch, seq_len, num_heads, head_dim]
            offset: Position offset for KV cache.

        Returns:
            Tensor with YaRN RoPE applied.
        """
        require_torch("YaRNRoPEMetal.apply")

        # Check if we can use Metal
        if self._lib is None or not x.is_mps:
            return super().apply(x, offset)

        # Compute attention scale (mscale)
        attention_scale = get_yarn_mscale(self.config.scale_factor, self.config.mscale_all_dim)
        if self.config.attention_factor is not None:
            attention_scale = self.config.attention_factor

        # Detect input layout
        # [batch, heads, seq, dim] vs [batch, seq, heads, dim]
        if x.dim() == 4:
            if x.shape[2] < x.shape[1]:
                # [batch, seq, heads, dim] - standard layout for our kernel
                x_transposed = x
                needs_transpose = False
            else:
                # [batch, heads, seq, dim] - transpose for kernel
                x_transposed = x.transpose(1, 2).contiguous()
                needs_transpose = True
        else:
            return super().apply(x, offset)

        try:
            output = dispatch_rope_yarn_forward(
                self._lib,
                x_transposed,
                self.cos_cache,
                self.sin_cache,
                attention_scale,
                offset,
            )

            if needs_transpose:
                output = output.transpose(1, 2).contiguous()

            return output.to(x.dtype)

        except Exception:
            # Fall back to PyTorch implementation
            return super().apply(x, offset)

    def apply_qk_fused(
        self,
        q: Any,
        k: Any,
        offset: int = 0,
    ) -> tuple[Any, Any]:
        """Apply YaRN RoPE to Q and K in a single fused kernel.

        More efficient than calling apply() twice when both Q and K need RoPE.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            offset: Position offset for KV cache.

        Returns:
            Tuple of (q_rotated, k_rotated).
        """
        require_torch("YaRNRoPEMetal.apply_qk_fused")

        # Compute attention scale
        attention_scale = get_yarn_mscale(self.config.scale_factor, self.config.mscale_all_dim)
        if self.config.attention_factor is not None:
            attention_scale = self.config.attention_factor

        if self._lib is None or not q.is_mps or not k.is_mps:
            # Fall back to separate applications
            return super().apply(q, offset), super().apply(k, offset)

        try:
            return dispatch_rope_yarn_qk_fused(
                self._lib,
                q,
                k,
                self.cos_cache,
                self.sin_cache,
                attention_scale,
                offset,
            )
        except Exception:
            return super().apply(q, offset), super().apply(k, offset)


class YaRNRoPELatent(YaRNRoPE):
    """YaRN RoPE for MLA latent representations.

    Specialized for DeepSeek MLA where the compressed KV representation
    contains both latent (kv_lora_rank) and positional (qk_rope_head_dim) parts.
    Only the positional part gets RoPE applied.

    Usage:
        rope = YaRNRoPELatent(
            rope_dim=64,           # qk_rope_head_dim
            kv_lora_rank=512,      # Latent dimension
            max_seq_len=131072,
            config=yarn_config,
        )
        kv_compressed = rope.apply(kv_a_proj_output, offset=0)
    """

    def __init__(
        self,
        rope_dim: int,
        kv_lora_rank: int,
        max_seq_len: int,
        base: float = 10000.0,
        config: YaRNConfig | None = None,
        device: str = "mps",
        dtype: Any = None,
    ):
        """Initialize YaRN RoPE for MLA latents.

        Args:
            rope_dim: RoPE dimension (qk_rope_head_dim).
            kv_lora_rank: Latent dimension (passes through unchanged).
            max_seq_len: Maximum sequence length.
            base: RoPE base frequency.
            config: YaRN configuration.
            device: Target device.
            dtype: Cache dtype.
        """
        # Use rope_dim for the parent class (this determines inv_freq/cache size)
        super().__init__(rope_dim, max_seq_len, base, config, device, dtype)
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim

        self._lib: Any | None = None
        if HAS_METAL_DISPATCH:
            try:
                self._lib = get_default_library()
            except Exception:
                pass

    def apply(
        self,
        x: Any,
        offset: int = 0,
    ) -> Any:
        """Apply YaRN RoPE to MLA latent representation.

        Args:
            x: Input tensor [batch, seq_len, kv_lora_rank + rope_dim]
            offset: Position offset for KV cache.

        Returns:
            Tensor with RoPE applied to positional portion only.
        """
        require_torch("YaRNRoPELatent.apply")

        batch_size, seq_len, total_dim = x.shape
        expected_dim = self.kv_lora_rank + self.rope_dim
        if total_dim != expected_dim:
            raise ValueError(f"Expected dim {expected_dim}, got {total_dim}")

        # Compute attention scale
        attention_scale = get_yarn_mscale(self.config.scale_factor, self.config.mscale_all_dim)
        if self.config.attention_factor is not None:
            attention_scale = self.config.attention_factor

        # Try Metal dispatch
        if self._lib is not None and x.is_mps:
            try:
                return dispatch_rope_yarn_latent(
                    self._lib,
                    x,
                    self.cos_cache,
                    self.sin_cache,
                    attention_scale,
                    self.kv_lora_rank,
                    self.rope_dim,
                    offset,
                )
            except Exception:
                pass

        # PyTorch fallback
        # Split into latent and positional parts
        latent = x[..., : self.kv_lora_rank]
        positional = x[..., self.kv_lora_rank :]

        # Get cos/sin for these positions
        cos = self.cos_cache[offset : offset + seq_len]
        sin = self.sin_cache[offset : offset + seq_len]

        # Reshape for broadcasting: [1, seq_len, dim//2]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Apply RoPE to positional part
        pos_even = positional[..., ::2]
        pos_odd = positional[..., 1::2]

        rotated_even = pos_even * cos - pos_odd * sin
        rotated_odd = pos_odd * cos + pos_even * sin

        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated = rotated.view(batch_size, seq_len, self.rope_dim) * attention_scale

        # Concatenate back
        return torch.cat([latent, rotated], dim=-1).to(x.dtype)
