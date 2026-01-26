"""Base types and protocols for hybrid architecture layers.

Defines the common interface that all layer types (Attention, Mamba, Hyena, etc.)
must implement to be used in hybrid models.

This module is framework-agnostic at runtime but provides PyTorch type hints
for static analysis. Tensor types use `Any` at runtime to support multiple
backends (PyTorch, NumPy) without hard dependencies.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from ..kv_cache import KVCache

    # Type alias for tensor-like objects (torch.Tensor or numpy.ndarray)
    TensorLike = torch.Tensor | Any


class HybridLayerType(Enum):
    """Supported layer types in hybrid architectures."""

    ATTENTION = auto()  # Standard transformer attention
    MAMBA = auto()  # Selective State Space (S6) from Mamba
    MAMBA2 = auto()  # Mamba-2 with SSD (Structured State Space Duality)
    HYENA = auto()  # Hyena operator (long convolutions)
    LINEAR_ATTENTION = auto()  # Linear attention variants (RWKV, RetNet)
    MLP_ONLY = auto()  # Pure MLP layer (no attention/SSM component)
    MOE = auto()  # Mixture of Experts layer
    SHARED_ATTENTION = auto()  # Attention with shared weights across layers (Zamba)


class StateType(Enum):
    """Type of state maintained by a layer across tokens."""

    NONE = auto()  # Stateless layer
    KV_CACHE = auto()  # Key-value cache (attention)
    SSM_STATE = auto()  # Continuous state (Mamba, RWKV)
    CONV_STATE = auto()  # Convolutional state buffer
    HYBRID = auto()  # Multiple state types


@dataclass
class LayerState:
    """Container for layer state during autoregressive generation.

    Different layer types maintain different state:
    - Attention: KV cache
    - Mamba: SSM hidden state + conv state
    - Hyena: Filter state

    This unified container allows the inference loop to manage state
    without knowing layer internals.

    Attributes:
        state_type: What kind of state this contains.
        layer_idx: Which layer this state belongs to.
        kv_cache: Optional KV cache for attention layers.
        ssm_state: Optional SSM state tensor [batch, d_state, d_model].
            Typically torch.Tensor on MPS/CUDA or numpy.ndarray.
        conv_state: Optional convolution state buffer [batch, d_conv, d_model].
            Typically torch.Tensor on MPS/CUDA or numpy.ndarray.
        metadata: Additional layer-specific state info.
    """

    state_type: StateType
    layer_idx: int
    kv_cache: KVCache | None = None
    ssm_state: Any | None = None  # torch.Tensor or numpy.ndarray
    conv_state: Any | None = None  # torch.Tensor or numpy.ndarray
    metadata: dict[str, Any] | None = None

    @classmethod
    def empty(cls, layer_idx: int, state_type: StateType) -> LayerState:
        """Create empty state for initialization."""
        return cls(state_type=state_type, layer_idx=layer_idx)

    @classmethod
    def for_attention(cls, layer_idx: int, kv_cache: KVCache) -> LayerState:
        """Create state container for attention layer."""
        return cls(
            state_type=StateType.KV_CACHE,
            layer_idx=layer_idx,
            kv_cache=kv_cache,
        )

    @classmethod
    def for_mamba(
        cls,
        layer_idx: int,
        ssm_state: Any,
        conv_state: Any | None = None,
    ) -> LayerState:
        """Create state container for Mamba layer."""
        return cls(
            state_type=StateType.SSM_STATE if conv_state is None else StateType.HYBRID,
            layer_idx=layer_idx,
            ssm_state=ssm_state,
            conv_state=conv_state,
        )

    def update_ssm(self, new_ssm_state: Any) -> LayerState:
        """Return new state with updated SSM state."""
        return LayerState(
            state_type=self.state_type,
            layer_idx=self.layer_idx,
            kv_cache=self.kv_cache,
            ssm_state=new_ssm_state,
            conv_state=self.conv_state,
            metadata=self.metadata,
        )

    def update_conv(self, new_conv_state: Any) -> LayerState:
        """Return new state with updated conv state."""
        return LayerState(
            state_type=self.state_type,
            layer_idx=self.layer_idx,
            kv_cache=self.kv_cache,
            ssm_state=self.ssm_state,
            conv_state=new_conv_state,
            metadata=self.metadata,
        )


@runtime_checkable
class LayerProtocol(Protocol):
    """Protocol that all hybrid layer implementations must satisfy.

    This allows the HybridBlock to dispatch to different layer types
    while maintaining a consistent interface for inference.

    Tensor arguments accept torch.Tensor (MPS/CUDA/CPU) or numpy.ndarray.
    """

    layer_type: HybridLayerType
    state_type: StateType
    hidden_size: int

    @abstractmethod
    def __call__(
        self,
        hidden_states: Any,  # torch.Tensor or numpy.ndarray
        position_ids: Any | None = None,
        state: LayerState | None = None,
        attention_mask: Any | None = None,
    ) -> tuple[Any, LayerState | None]:
        """Forward pass through the layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
                Can be torch.Tensor (MPS/CUDA/CPU) or numpy.ndarray.
            position_ids: Optional position IDs for position-dependent layers.
            state: Optional layer state from previous timestep.
            attention_mask: Optional mask for attention layers.

        Returns:
            Tuple of (output tensor, updated state or None).
        """
        ...

    @abstractmethod
    def init_state(self, batch_size: int, layer_idx: int) -> LayerState:
        """Initialize state for autoregressive generation.

        Args:
            batch_size: Batch dimension for state tensors.
            layer_idx: Index of this layer in the model.

        Returns:
            Initialized LayerState appropriate for this layer type.
        """
        ...


def get_layer_state_size(
    layer_type: HybridLayerType,
    hidden_size: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> dict[str, int]:
    """Estimate state memory requirements for a layer type.

    Useful for memory planning in hybrid models where different layers
    have very different state requirements.

    Args:
        layer_type: Type of layer.
        hidden_size: Model hidden dimension.
        d_state: SSM state dimension (for Mamba).
        d_conv: Convolution width (for Mamba).
        expand: Expansion factor for inner dimension.

    Returns:
        Dict with state component sizes in bytes (assuming float16).
    """
    inner_dim = hidden_size * expand

    if layer_type == HybridLayerType.ATTENTION:
        # KV cache per token (grows with sequence length)
        # Per position: 2 * hidden_size * 2 bytes
        return {
            "per_token_bytes": 2 * hidden_size * 2,  # K and V
            "fixed_bytes": 0,
            "grows_with_seq": True,
        }

    elif layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2):
        # Fixed SSM state + conv buffer
        ssm_bytes = d_state * inner_dim * 2  # float16
        conv_bytes = d_conv * inner_dim * 2  # float16
        return {
            "per_token_bytes": 0,
            "fixed_bytes": ssm_bytes + conv_bytes,
            "grows_with_seq": False,
        }

    elif layer_type == HybridLayerType.LINEAR_ATTENTION:
        # Linear attention has fixed-size state
        # State: [d_k, d_v] matrix
        state_bytes = hidden_size * hidden_size // 4 * 2  # Approximate
        return {
            "per_token_bytes": 0,
            "fixed_bytes": state_bytes,
            "grows_with_seq": False,
        }

    elif layer_type == HybridLayerType.HYENA:
        # Hyena has filter state
        filter_bytes = hidden_size * 2
        return {
            "per_token_bytes": 0,
            "fixed_bytes": filter_bytes,
            "grows_with_seq": False,
        }

    else:
        # Stateless layers
        return {
            "per_token_bytes": 0,
            "fixed_bytes": 0,
            "grows_with_seq": False,
        }


def estimate_hybrid_state_memory(
    layer_types: list[HybridLayerType],
    hidden_size: int,
    batch_size: int,
    max_seq_len: int,
    d_state: int = 16,
    d_conv: int = 4,
) -> dict[str, int]:
    """Estimate total state memory for a hybrid model.

    Args:
        layer_types: List of layer types in order.
        hidden_size: Model hidden dimension.
        batch_size: Batch size for inference.
        max_seq_len: Maximum sequence length.
        d_state: SSM state dimension.
        d_conv: Convolution width.

    Returns:
        Dict with memory estimates in bytes.
    """
    total_fixed = 0
    total_per_seq = 0

    for layer_type in layer_types:
        state_info = get_layer_state_size(layer_type, hidden_size, d_state, d_conv)
        total_fixed += state_info["fixed_bytes"]
        if state_info["grows_with_seq"]:
            total_per_seq += state_info["per_token_bytes"]

    return {
        "fixed_state_bytes": total_fixed * batch_size,
        "kv_cache_bytes": total_per_seq * batch_size * max_seq_len,
        "total_bytes": (total_fixed * batch_size + total_per_seq * batch_size * max_seq_len),
        "num_attention_layers": sum(1 for t in layer_types if t == HybridLayerType.ATTENTION),
        "num_ssm_layers": sum(
            1 for t in layer_types if t in (HybridLayerType.MAMBA, HybridLayerType.MAMBA2)
        ),
    }
