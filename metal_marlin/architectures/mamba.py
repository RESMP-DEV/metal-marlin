"""Mamba (Selective State Space) layer implementation for hybrid models.

Implements the Mamba architecture from:
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752

Key components:
- Selective scan (S6) - input-dependent SSM parameters
- Convolutional branch for local context
- Gated output projection

This implementation supports quantized weights via MarlinLinear for
memory-efficient inference on Apple Silicon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._compat import HAS_TORCH, require_torch, torch
from ..layers import MarlinLinear
from ..mixed_precision import LayerQuantConfig, Precision
from .base import HybridLayerType, LayerState, StateType

if TYPE_CHECKING:
    import torch as torch_typing


@dataclass
class MambaState:
    """State container for Mamba layer during autoregressive generation.

    Attributes:
        ssm_state: [batch, d_state, d_inner] - Hidden state of the SSM.
        conv_state: [batch, d_conv - 1, d_inner] - Convolutional state buffer.
    """

    ssm_state: Any  # torch.Tensor when PyTorch available
    conv_state: Any  # torch.Tensor when PyTorch available

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        d_state: int,
        d_inner: int,
        d_conv: int,
        device: str = "mps",
        dtype: Any = None,
    ) -> MambaState:
        """Create zero-initialized state."""
        require_torch("Mamba state initialization")
        assert torch is not None

        if dtype is None:
            dtype = torch.float32

        return cls(
            ssm_state=torch.zeros((batch_size, d_state, d_inner), device=device, dtype=dtype),
            conv_state=torch.zeros((batch_size, d_conv - 1, d_inner), device=device, dtype=dtype),
        )


@dataclass
class SelectiveScanConfig:
    """Configuration for selective scan operation.

    Attributes:
        d_state: State dimension (N).
        dt_rank: Rank of delta projection.
        A_init: Method for A matrix initialization.
        dt_min: Minimum delta value.
        dt_max: Maximum delta value.
        dt_floor: Floor for delta initialization.
    """

    d_state: int = 16
    dt_rank: int | str = "auto"
    A_init: str = "random"  # "random", "S4D", or "hippo"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_floor: float = 1e-4


def _get_base_class() -> type:
    """Return torch.nn.Module if PyTorch is available, else object."""
    if HAS_TORCH and torch is not None:
        return torch.nn.Module
    return object


class _MambaBlockBase:
    """Base implementation of Mamba block without nn.Module inheritance.

    Architecture:
        x -> Linear(in_proj) -> split -> [xz, x_proj]
                                            |
        x_proj -> Conv1D -> SiLU -> SSM(A, B, C, dt) -> * z -> Linear(out_proj) -> out

    The selective scan allows the model to selectively remember or forget
    information based on the input, unlike traditional RNNs with fixed dynamics.
    """

    layer_type = HybridLayerType.MAMBA
    state_type = StateType.HYBRID  # Both SSM and conv state

    hidden_size: int
    d_state: int
    d_conv: int
    expand: int
    d_inner: int
    dt_rank: int
    conv_bias: bool
    bias: bool
    quant_config: LayerQuantConfig
    device: str

    # Layers
    in_proj: MarlinLinear
    conv1d: Any  # torch.nn.Conv1d
    x_proj: MarlinLinear
    dt_proj: Any  # torch.nn.Linear
    out_proj: MarlinLinear

    # Parameters
    A_log: Any  # torch.nn.Parameter
    D: Any  # torch.nn.Parameter

    def _init_mamba(
        self,
        hidden_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        quant_config: LayerQuantConfig | None = None,
        device: str = "mps",
    ) -> None:
        """Initialize Mamba block.

        Args:
            hidden_size: Input/output dimension.
            d_state: State dimension for SSM (N in paper).
            d_conv: Convolutional kernel width.
            expand: Expansion factor for inner dimension.
            dt_rank: Rank for delta projection. "auto" = ceil(hidden_size / 16).
            conv_bias: Use bias in convolution.
            bias: Use bias in linear projections.
            quant_config: Quantization settings for linear layers.
            device: Device for tensor allocation.
        """
        require_torch("MambaBlock")
        assert torch is not None

        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = hidden_size * expand
        self.device = device

        # Compute dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(hidden_size / 16)
        else:
            self.dt_rank = int(dt_rank)

        self.conv_bias = conv_bias
        self.bias = bias
        self.quant_config = quant_config or LayerQuantConfig(Precision.FP4_E2M1, 128)

        # Input projection: maps hidden_size -> 2 * d_inner
        # Split into: x (for conv/SSM) and z (for gating)
        self.in_proj = MarlinLinear(
            hidden_size,
            2 * self.d_inner,
            bias=bias,
            quant_type=self._get_quant_type(),
            group_size=self.quant_config.group_size,
        )

        # 1D convolution for local context
        # Note: For quantized inference, we keep conv in FP16 (small parameter count)
        self.conv1d = torch.nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise
            bias=conv_bias,
            padding=d_conv - 1,
            device=device,
        )

        # x_proj: maps d_inner -> dt_rank + 2*d_state (for B and C)
        # Small projection, can use higher precision
        self.x_proj = MarlinLinear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
            quant_type=self._get_quant_type(),
            group_size=min(64, self.quant_config.group_size),
        )

        # dt_proj: maps dt_rank -> d_inner
        self.dt_proj = torch.nn.Linear(self.dt_rank, self.d_inner, bias=True, device=device)

        # A and D parameters (not quantized - small)
        # A is the state transition matrix (initialized for stability)
        self.A_log = torch.nn.Parameter(torch.zeros((d_state, self.d_inner), device=device))
        self.D = torch.nn.Parameter(torch.ones((self.d_inner,), device=device))

        # Output projection: maps d_inner -> hidden_size
        self.out_proj = MarlinLinear(
            self.d_inner,
            hidden_size,
            bias=bias,
            quant_type=self._get_quant_type(),
            group_size=self.quant_config.group_size,
        )

        # Initialize parameters
        self._init_parameters()

    def _get_quant_type(self) -> str:
        """Convert precision enum to string for MarlinLinear."""
        precision_map = {
            Precision.FP4_E2M1: "fp4",
            Precision.INT4: "int4",
            Precision.FP8_E4M3: "fp8",
            Precision.INT8: "int8",
            Precision.FP16: "fp16",
            Precision.BF16: "bf16",
        }
        return precision_map.get(self.quant_config.precision, "fp4")

    def _init_parameters(self) -> None:
        """Initialize SSM parameters for stability."""
        assert torch is not None

        # A initialization: use S4D-Real scheme for stability
        # A = -exp(uniform(log(dt_min), log(dt_max)))
        # This gives a distribution of timescales
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device)
        A = A.unsqueeze(1).expand(self.d_state, self.d_inner)
        self.A_log.data = torch.log(A)

        # D is a skip connection (residual)
        self.D.data = torch.ones((self.d_inner,), device=self.device)

        # Initialize dt_proj bias to encourage small initial dt
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            # Initialize to values that give dt in [dt_min, dt_max]
            dt_init = torch.exp(
                torch.empty(self.d_inner, device=self.device).uniform_(
                    math.log(0.001), math.log(0.1)
                )
            )
            # Inverse of softplus to get bias
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)

    def _forward(
        self,
        hidden_states: torch_typing.Tensor,
        position_ids: torch_typing.Tensor | None = None,
        state: LayerState | None = None,
        attention_mask: torch_typing.Tensor | None = None,
    ) -> tuple[torch_typing.Tensor, LayerState | None]:
        """Forward pass through Mamba block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: Unused (Mamba is position-agnostic internally).
            state: Optional state from previous timestep.
            attention_mask: Unused for Mamba.

        Returns:
            (output, new_state) where output is [batch, seq_len, hidden_size]
        """
        assert torch is not None
        batch_size, seq_len, _ = hidden_states.shape

        # Project input to (x, z) pair
        xz = self.in_proj(hidden_states)  # [B, L, 2*d_inner]
        x, z = torch.chunk(xz, 2, dim=-1)  # Each [B, L, d_inner]

        # Convolutional branch with state handling
        if state is not None and state.conv_state is not None:
            # Prepend conv state for causal convolution
            conv_state = state.conv_state  # [B, d_conv-1, d_inner]
            x_with_state = torch.cat([conv_state, x.transpose(1, 2)], dim=2)
            # [B, d_inner, L+d_conv-1]
            x_transposed = x_with_state.transpose(1, 2)  # [B, L+d_conv-1, d_inner]
            conv_out = self.conv1d(x_transposed.transpose(1, 2))[:, :, :seq_len]
            conv_out = conv_out.transpose(1, 2)  # [B, L, d_inner]
            # Update conv state (last d_conv-1 positions)
            new_conv_state = x.transpose(1, 2)[:, :, -(self.d_conv - 1) :]
        else:
            # Transpose for conv1d: [B, d_inner, L]
            x_transposed = x.transpose(1, 2)
            conv_out = self.conv1d(x_transposed)[:, :, :seq_len]
            conv_out = conv_out.transpose(1, 2)  # Back to [B, L, d_inner]
            new_conv_state = (
                x.transpose(1, 2)[:, :, -(self.d_conv - 1) :]
                if seq_len >= self.d_conv - 1
                else None
            )

        # Apply activation (SiLU/Swish)
        x = torch.nn.functional.silu(conv_out)

        # Compute SSM parameters from input (selective!)
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]

        # Split into dt, B, C
        dt = x_dbl[:, :, : self.dt_rank]  # [B, L, dt_rank]
        B = x_dbl[:, :, self.dt_rank : self.dt_rank + self.d_state]  # [B, L, d_state]
        C = x_dbl[:, :, self.dt_rank + self.d_state :]  # [B, L, d_state]

        # Project dt to d_inner and apply softplus
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = torch.nn.functional.softplus(dt)  # Ensure positive

        # Get A from log
        A = -torch.exp(self.A_log)  # [d_state, d_inner]

        # Run selective scan
        if state is not None and state.ssm_state is not None:
            ssm_state = state.ssm_state
        else:
            ssm_state = torch.zeros(
                (batch_size, self.d_state, self.d_inner),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        y, new_ssm_state = self._selective_scan(x, dt, A, B, C, ssm_state)

        # Gate with z
        y = y * torch.sigmoid(z)

        # Skip connection with D
        y = y + self.D * x

        # Project to output
        output = self.out_proj(y)

        # Build new state
        new_state = LayerState.for_mamba(
            layer_idx=state.layer_idx if state else 0,
            ssm_state=new_ssm_state,
            conv_state=new_conv_state,
        )

        return output, new_state

    def _selective_scan(
        self,
        x: torch_typing.Tensor,  # [B, L, d_inner]
        dt: torch_typing.Tensor,  # [B, L, d_inner]
        A: torch_typing.Tensor,  # [d_state, d_inner]
        B: torch_typing.Tensor,  # [B, L, d_state]
        C: torch_typing.Tensor,  # [B, L, d_state]
        ssm_state: torch_typing.Tensor,  # [B, d_state, d_inner]
    ) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
        """Selective scan operation (sequential version).

        This is the key innovation of Mamba: the SSM parameters (A, B, C)
        are computed from the input, allowing selective memory.

        For parallel training, an efficient scan operation is used.
        For inference, we unroll sequentially (fast for short sequences).

        Args:
            x: Input sequence.
            dt: Time step (delta) per position.
            A: State transition matrix (shared across positions).
            B: Input-dependent input matrix.
            C: Input-dependent output matrix.
            ssm_state: Initial SSM state.

        Returns:
            (output_sequence, final_ssm_state)
        """
        assert torch is not None
        batch_size, seq_len, d_inner = x.shape

        # Discretize A with zero-order hold: A_bar = exp(dt * A)
        # Note: A is [d_state, d_inner], dt is [B, L, d_inner]
        # We need A_bar as [B, L, d_state, d_inner]
        dt_A = dt.unsqueeze(2) * A.unsqueeze(0).unsqueeze(0)  # [B, L, d_state, d_inner]
        A_bar = torch.exp(dt_A)

        # Discretize B with zero-order hold: B_bar = (A^-1 * (A_bar - I)) * B
        # Simplified: B_bar ≈ dt * B (when dt is small)
        # B is [B, L, d_state], dt is [B, L, d_inner]
        # Need B_bar as [B, L, d_state, d_inner]
        B_bar = dt.unsqueeze(2) * B.unsqueeze(-1)  # [B, L, d_state, d_inner]

        # Sequential scan
        outputs = []
        h = ssm_state  # [B, d_state, d_inner]

        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            A_bar_t = A_bar[:, t, :, :]  # [B, d_state, d_inner]
            B_bar_t = B_bar[:, t, :, :]  # [B, d_state, d_inner]
            x_t = x[:, t, :]  # [B, d_inner]

            h = A_bar_t * h + B_bar_t * x_t.unsqueeze(1)  # [B, d_state, d_inner]

            # Output: y = C * h
            C_t = C[:, t, :]  # [B, d_state]
            y_t = torch.sum(C_t.unsqueeze(-1) * h, dim=1)  # [B, d_inner]

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y, h

    def _init_state(self, batch_size: int, layer_idx: int) -> LayerState:
        """Initialize state for autoregressive generation."""
        mamba_state = MambaState.zeros(
            batch_size=batch_size,
            d_state=self.d_state,
            d_inner=self.d_inner,
            d_conv=self.d_conv,
            device=self.device,
        )
        return LayerState(
            state_type=StateType.HYBRID,
            layer_idx=layer_idx,
            ssm_state=mamba_state.ssm_state,
            conv_state=mamba_state.conv_state,
        )


class _MarlinMambaBlockBase:
    """Base implementation of full Mamba block with input/output norms and MLP.

    Matches the structure expected by hybrid models:
        x -> Norm -> Mamba -> residual -> Norm -> MLP -> residual

    This mirrors MarlinTransformerBlock but uses Mamba instead of attention.
    """

    layer_type = HybridLayerType.MAMBA
    state_type = StateType.HYBRID

    hidden_size: int
    intermediate_size: int
    mamba: Any  # MambaBlock
    input_layernorm: Any  # RMSNorm
    post_mamba_layernorm: Any  # RMSNorm
    mlp: Any  # MarlinMLP

    def _init_marlin_mamba(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        quant_config: LayerQuantConfig | None = None,
        mlp_quant_config: LayerQuantConfig | None = None,
        rms_norm_eps: float = 1e-6,
        use_gated_mlp: bool = True,
        mlp_activation: str = "silu",
        device: str = "mps",
    ) -> None:
        """Initialize full Mamba block.

        Args:
            hidden_size: Model hidden dimension.
            intermediate_size: MLP intermediate size. Default 4 * hidden_size.
            d_state: SSM state dimension.
            d_conv: Convolution width.
            expand: Expansion factor for Mamba inner dimension.
            quant_config: Quantization for Mamba projections.
            mlp_quant_config: Quantization for MLP. Default same as quant_config.
            rms_norm_eps: Epsilon for RMSNorm.
            use_gated_mlp: Use gated MLP (SwiGLU).
            mlp_activation: Activation for MLP.
            device: Device for tensor allocation.
        """
        require_torch("MarlinMambaBlock")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4

        # Mamba block
        self.mamba = MambaBlock(
            hidden_size=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            quant_config=quant_config,
            device=device,
        )

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, device=device)
        self.post_mamba_layernorm = RMSNorm(hidden_size, rms_norm_eps, device=device)

        # MLP (import here to avoid circular)
        from ..mlp import MarlinMLP

        mlp_quant = mlp_quant_config or quant_config
        quant_type = "fp4" if mlp_quant is None else self._get_quant_type_mlp(mlp_quant)
        group_size = 128 if mlp_quant is None else mlp_quant.group_size

        self.mlp = MarlinMLP(
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            quant_type=quant_type,
            group_size=group_size,
            activation=mlp_activation,
            gated=use_gated_mlp,
        )

    def _get_quant_type_mlp(self, config: LayerQuantConfig) -> str:
        """Convert precision enum to string."""
        precision_map = {
            Precision.FP4_E2M1: "fp4",
            Precision.INT4: "int4",
            Precision.FP8_E4M3: "fp8",
            Precision.INT8: "int8",
            Precision.FP16: "fp16",
            Precision.BF16: "bf16",
        }
        return precision_map.get(config.precision, "fp4")

    def _forward_block(
        self,
        hidden_states: torch_typing.Tensor,
        position_ids: torch_typing.Tensor | None = None,
        state: LayerState | None = None,
        attention_mask: torch_typing.Tensor | None = None,
    ) -> tuple[torch_typing.Tensor, LayerState | None]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: Unused for Mamba.
            state: Optional Mamba state from previous timestep.
            attention_mask: Unused for Mamba.

        Returns:
            (output, new_state)
        """
        # Mamba with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_state = self.mamba(
            hidden_states,
            position_ids=position_ids,
            state=state,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_mamba_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_state

    def _init_state_block(self, batch_size: int, layer_idx: int) -> LayerState:
        """Initialize state for autoregressive generation."""
        return self.mamba.init_state(batch_size, layer_idx)


class _RMSNormBase:
    """Base RMSNorm implementation for Mamba blocks."""

    weight: Any  # torch.nn.Parameter
    eps: float

    def _init_rmsnorm(self, hidden_size: int, eps: float = 1e-6, device: str = "mps") -> None:
        require_torch("RMSNorm")
        assert torch is not None
        self.weight = torch.nn.Parameter(torch.ones((hidden_size,), device=device))
        self.eps = eps

    def _forward_rmsnorm(self, x: torch_typing.Tensor) -> torch_typing.Tensor:
        assert torch is not None
        variance = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# Conditionally define classes based on PyTorch availability
if HAS_TORCH and torch is not None:

    class MambaBlock(torch.nn.Module, _MambaBlockBase):
        """Mamba block with selective state space mechanism.

        Architecture:
            x -> Linear(in_proj) -> split -> [xz, x_proj]
                                                |
            x_proj -> Conv1D -> SiLU -> SSM(A, B, C, dt) -> * z -> Linear(out_proj) -> out

        The selective scan allows the model to selectively remember or forget
        information based on the input, unlike traditional RNNs with fixed dynamics.
        """

        def __init__(
            self,
            hidden_size: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dt_rank: int | str = "auto",
            conv_bias: bool = True,
            bias: bool = False,
            quant_config: LayerQuantConfig | None = None,
            device: str = "mps",
        ):
            """Initialize Mamba block.

            Args:
                hidden_size: Input/output dimension.
                d_state: State dimension for SSM (N in paper).
                d_conv: Convolutional kernel width.
                expand: Expansion factor for inner dimension.
                dt_rank: Rank for delta projection. "auto" = ceil(hidden_size / 16).
                conv_bias: Use bias in convolution.
                bias: Use bias in linear projections.
                quant_config: Quantization settings for linear layers.
                device: Device for tensor allocation.
            """
            torch.nn.Module.__init__(self)
            self._init_mamba(
                hidden_size,
                d_state,
                d_conv,
                expand,
                dt_rank,
                conv_bias,
                bias,
                quant_config,
                device,
            )

        def forward(
            self,
            hidden_states: torch_typing.Tensor,
            position_ids: torch_typing.Tensor | None = None,
            state: LayerState | None = None,
            attention_mask: torch_typing.Tensor | None = None,
        ) -> tuple[torch_typing.Tensor, LayerState | None]:
            return self._forward(hidden_states, position_ids, state, attention_mask)

        def init_state(self, batch_size: int, layer_idx: int) -> LayerState:
            return self._init_state(batch_size, layer_idx)

    class MarlinMambaBlock(torch.nn.Module, _MarlinMambaBlockBase):
        """Full Mamba block with input/output norms and MLP.

        Matches the structure expected by hybrid models:
            x -> Norm -> Mamba -> residual -> Norm -> MLP -> residual

        This mirrors MarlinTransformerBlock but uses Mamba instead of attention.
        """

        def __init__(
            self,
            hidden_size: int,
            intermediate_size: int | None = None,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            quant_config: LayerQuantConfig | None = None,
            mlp_quant_config: LayerQuantConfig | None = None,
            rms_norm_eps: float = 1e-6,
            use_gated_mlp: bool = True,
            mlp_activation: str = "silu",
            device: str = "mps",
        ):
            """Initialize full Mamba block.

            Args:
                hidden_size: Model hidden dimension.
                intermediate_size: MLP intermediate size. Default 4 * hidden_size.
                d_state: SSM state dimension.
                d_conv: Convolution width.
                expand: Expansion factor for Mamba inner dimension.
                quant_config: Quantization for Mamba projections.
                mlp_quant_config: Quantization for MLP. Default same as quant_config.
                rms_norm_eps: Epsilon for RMSNorm.
                use_gated_mlp: Use gated MLP (SwiGLU).
                mlp_activation: Activation for MLP.
                device: Device for tensor allocation.
            """
            torch.nn.Module.__init__(self)
            self._init_marlin_mamba(
                hidden_size,
                intermediate_size,
                d_state,
                d_conv,
                expand,
                quant_config,
                mlp_quant_config,
                rms_norm_eps,
                use_gated_mlp,
                mlp_activation,
                device,
            )

        def forward(
            self,
            hidden_states: torch_typing.Tensor,
            position_ids: torch_typing.Tensor | None = None,
            state: LayerState | None = None,
            attention_mask: torch_typing.Tensor | None = None,
        ) -> tuple[torch_typing.Tensor, LayerState | None]:
            return self._forward_block(hidden_states, position_ids, state, attention_mask)

        def init_state(self, batch_size: int, layer_idx: int) -> LayerState:
            return self._init_state_block(batch_size, layer_idx)

    class RMSNorm(torch.nn.Module, _RMSNormBase):
        """RMSNorm for Mamba blocks."""

        def __init__(self, hidden_size: int, eps: float = 1e-6, device: str = "mps"):
            torch.nn.Module.__init__(self)
            self._init_rmsnorm(hidden_size, eps, device)

        def forward(self, x: torch_typing.Tensor) -> torch_typing.Tensor:
            return self._forward_rmsnorm(x)

else:

    class MambaBlock(_MambaBlockBase):  # type: ignore[no-redef]
        """Mamba block stub when PyTorch is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MambaBlock requires PyTorch for inference. Install PyTorch with: pip install torch"
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("MambaBlock requires PyTorch")

        def init_state(self, batch_size: int, layer_idx: int) -> LayerState:
            raise RuntimeError("MambaBlock requires PyTorch")

    class MarlinMambaBlock(_MarlinMambaBlockBase):  # type: ignore[no-redef]
        """MarlinMambaBlock stub when PyTorch is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MarlinMambaBlock requires PyTorch for inference. "
                "Install PyTorch with: pip install torch"
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("MarlinMambaBlock requires PyTorch")

        def init_state(self, batch_size: int, layer_idx: int) -> LayerState:
            raise RuntimeError("MarlinMambaBlock requires PyTorch")

    class RMSNorm(_RMSNormBase):  # type: ignore[no-redef]
        """RMSNorm stub when PyTorch is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("RMSNorm requires PyTorch. Install PyTorch with: pip install torch")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("RMSNorm requires PyTorch")


__all__ = [
    "MambaBlock",
    "MambaState",
    "MarlinMambaBlock",
    "RMSNorm",
    "SelectiveScanConfig",
]
