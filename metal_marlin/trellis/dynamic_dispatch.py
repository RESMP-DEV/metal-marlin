"""Dynamic dispatch for mixed-precision MoE layers.

Provides C++-backed dynamic dispatch for MoE layers with varying expert
bit widths. Automatically selects optimal kernels based on per-expert
precision configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..metal_dispatch import HAS_METAL, MetalKernelLibrary, get_fast_path

if TYPE_CHECKING:
    from .moe_dispatch import CachedWeightBuffers, MoEBufferPool

logger = logging.getLogger(__name__)

_MANAGED_BUFFER_TYPENAME = "ManagedBuffer"
_HALF_COMPATIBLE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)
_INT32_COMPATIBLE_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_UINT8_COMPATIBLE_DTYPES = (
    torch.uint8,
    torch.int8,
)


def _normalize_tensor_for_cpp_bridge(
    tensor: Any,
    *,
    field_name: str,
    tensor_attr: str,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
    allowed_dtypes: tuple[torch.dtype, ...],
) -> torch.Tensor:
    """Normalize tensors before wrapping as nanobind ManagedBuffer."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Unsupported input for C++ bridge: "
            f"`weight_buffers.{tensor_attr}` for field `{field_name}` must be "
            f"a torch.Tensor, got {type(tensor).__name__}"
        )

    if tensor.layout is not torch.strided:
        raise TypeError(
            f"Unsupported tensor layout for C++ bridge field `{field_name}`: "
            f"`weight_buffers.{tensor_attr}` has layout={tensor.layout}. "
            "Only dense strided tensors are supported."
        )

    if tensor.dtype not in allowed_dtypes:
        allowed = ", ".join(str(dtype) for dtype in allowed_dtypes)
        raise TypeError(
            f"Unsupported dtype for C++ bridge field `{field_name}`: "
            f"`weight_buffers.{tensor_attr}` has dtype={tensor.dtype}. "
            f"Expected one of: {allowed}."
        )

    normalized = tensor
    try:
        if normalized.device != expected_device:
            normalized = normalized.to(device=expected_device)
        if normalized.dtype != expected_dtype:
            normalized = normalized.to(dtype=expected_dtype)
        if not normalized.is_contiguous():
            normalized = normalized.contiguous()
    except Exception as exc:  # pragma: no cover - depends on runtime/device state
        raise RuntimeError(
            f"Failed to normalize `weight_buffers.{tensor_attr}` for field "
            f"`{field_name}` (target device={expected_device}, "
            f"target dtype={expected_dtype}, contiguous=True): {exc}"
        ) from exc

    if normalized.device != expected_device:
        raise RuntimeError(
            f"Normalization failed for field `{field_name}`: expected device "
            f"{expected_device}, got {normalized.device}."
        )
    if normalized.dtype != expected_dtype:
        raise RuntimeError(
            f"Normalization failed for field `{field_name}`: expected dtype "
            f"{expected_dtype}, got {normalized.dtype}."
        )
    if not normalized.is_contiguous():
        raise RuntimeError(
            f"Normalization failed for field `{field_name}`: tensor is not contiguous."
        )

    return normalized


@dataclass
class ExpertBits:
    """Bit width configuration for a single expert.

    Attributes:
        gate_bits: Bit width for gate projection (2, 3, 4, etc.)
        up_bits: Bit width for up projection (2, 3, 4, etc.)
        down_bits: Bit width for down projection (2, 3, 4, etc.)
    """
    gate_bits: int
    up_bits: int
    down_bits: int


def is_dynamic_moe_available() -> bool:
    """Check if C++ dynamic MoE dispatch is available.

    Returns:
        True if the C++ extension is available and Metal is enabled.
    """
    if not HAS_METAL:
        return False

    try:
        # Try to import the C++ extension
        import metal_marlin._dynamic_moe  # type: ignore
        return True
    except ImportError:
        return False


class DynamicMoEDispatch:
    """C++ dynamic dispatch for mixed-precision MoE layers.

    This class provides a high-level interface to the C++ dynamic dispatch
    implementation, which efficiently handles experts with varying bit widths
    by selecting optimal kernels at runtime.

    Args:
        lib: MetalKernelLibrary instance for kernel dispatch
        expert_bits: List of ExpertBits for each expert

    Example:
        >>> expert_bits = [
        ...     ExpertBits(gate_bits=4, up_bits=4, down_bits=4),
        ...     ExpertBits(gate_bits=3, up_bits=3, down_bits=3),
        ...     # ... more experts
        ... ]
        >>> dispatcher = DynamicMoEDispatch(lib, expert_bits)
        >>> output = dispatcher.dispatch(
        ...     activations=x,
        ...     expert_ids=selected_experts,
        ...     expert_probs=routing_weights,
        ...     weight_buffers=cached_buffers,
        ... )
    """

    def __init__(
        self,
        lib: MetalKernelLibrary,
        expert_bits: list[ExpertBits],
    ):
        """Initialize dynamic MoE dispatcher.

        Args:
            lib: MetalKernelLibrary instance
            expert_bits: List of ExpertBits configurations per expert

        Raises:
            RuntimeError: If dynamic dispatch is not available
        """
        if not is_dynamic_moe_available():
            raise RuntimeError(
                "C++ dynamic MoE dispatch not available. "
                "Ensure metal_marlin C++ extensions are built with Metal support."
            )

        self._lib = lib
        self._expert_bits = expert_bits
        self._num_experts = len(expert_bits)

        # Build bit width lookup tables for fast dispatch
        self._build_bit_lookup_tables()

        # Per-device cache of group expert-id tensors to avoid re-allocation
        # Structure: {device_str: {bit_tuple: tensor}}
        self._group_expert_tensor_cache: dict[
            str, dict[tuple[int, int, int], torch.Tensor]
        ] = {}

        logger.debug(
            f"Initialized DynamicMoEDispatch for {self._num_experts} experts "
            f"with mixed precision"
        )

    def _build_bit_lookup_tables(self) -> None:
        """Build lookup tables for fast bit width resolution."""
        # Extract bit widths into tensor format for GPU access
        self._gate_bits = torch.tensor(
            [eb.gate_bits for eb in self._expert_bits],
            dtype=torch.int32,
        )
        self._up_bits = torch.tensor(
            [eb.up_bits for eb in self._expert_bits],
            dtype=torch.int32,
        )
        self._down_bits = torch.tensor(
            [eb.down_bits for eb in self._expert_bits],
            dtype=torch.int32,
        )

        # Group experts by (gate, up, down) bit tuple for batching
        self._bit_groups: dict[tuple[int, int, int], list[int]] = {}
        for i, eb in enumerate(self._expert_bits):
            bit_tuple = (eb.gate_bits, eb.up_bits, eb.down_bits)
            if bit_tuple not in self._bit_groups:
                self._bit_groups[bit_tuple] = []
            self._bit_groups[bit_tuple].append(i)

        logger.debug(
            f"Built bit lookup tables: {len(self._bit_groups)} unique bit groups"
        )

    def dispatch(
        self,
        activations: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        gate_weights: torch.Tensor | None = None,
        gate_scales: torch.Tensor | None = None,
        up_weights: torch.Tensor | None = None,
        up_scales: torch.Tensor | None = None,
        down_weights: torch.Tensor | None = None,
        down_scales: torch.Tensor | None = None,
        gate_su: torch.Tensor | None = None,
        gate_sv: torch.Tensor | None = None,
        up_su: torch.Tensor | None = None,
        up_sv: torch.Tensor | None = None,
        down_su: torch.Tensor | None = None,
        down_sv: torch.Tensor | None = None,
        hidden_dim: int | None = None,
        intermediate_dim: int | None = None,
        num_experts: int | None = None,
        top_k: int | None = None,
        weight_buffers: CachedWeightBuffers | None = None,
        buffer_pool: MoEBufferPool | None = None,
        use_fp32_acc: bool = False,
    ) -> torch.Tensor:
        """Dispatch activations to experts using dynamic bit-width dispatch.

        This method routes each token to its selected experts and executes
        the MoE computation with per-expert bit width selection.

        Args:
            activations: Input tensor [batch, hidden_dim] in fp16
            expert_ids: Selected expert indices [batch, top_k]
            expert_probs: Normalized routing weights [batch, top_k]
            gate_weights: Packed gate projection weights (optional)
            gate_scales: Gate projection scales (optional)
            up_weights: Packed up projection weights (optional)
            up_scales: Up projection scales (optional)
            down_weights: Packed down projection weights (optional)
            down_scales: Down projection scales (optional)
            gate_su/gate_sv: Gate projection sign vectors (optional)
            up_su/up_sv: Up projection sign vectors (optional)
            down_su/down_sv: Down projection sign vectors (optional)
            hidden_dim: Hidden dimension size (inferred if None)
            intermediate_dim: Intermediate dimension size (inferred if None)
            num_experts: Number of experts (inferred from expert_bits if None)
            top_k: Number of experts per token (inferred from expert_ids if None)
            weight_buffers: Cached Metal buffers for expert weights
            buffer_pool: Optional buffer pool for intermediate allocations
            use_fp32_acc: Whether to use FP32 accumulation for better precision

        Returns:
            Output tensor [batch, hidden_dim] in fp16

        Raises:
            RuntimeError: If dispatch fails
        """
        import os
        import time

        batch_size = activations.shape[0]
        if top_k is None:
            top_k = expert_ids.shape[1]

        # Infer dimensions if not provided
        if hidden_dim is None:
            hidden_dim = activations.shape[-1]
        if intermediate_dim is None:
            # Try to infer from weight buffers or use common default
            intermediate_dim = hidden_dim * 4  # Common MLP expansion ratio
        if num_experts is None:
            num_experts = self._num_experts

        # Ensure weight buffers are in nanobind-compatible format for C++ extension
        # The C++ extension (_cpp_ext) expects ManagedBuffer objects, not MTLBuffer
        if weight_buffers is not None:
            fast_path = get_fast_path(self._lib)
            if fast_path.available:
                # Maintain a persistent list of tensors to keep them alive for C++.
                # This is critical as ManagedBuffer objects hold raw pointers to them.
                if not hasattr(weight_buffers, "_managed_tensors"):
                    weight_buffers._managed_tensors = []
                elif not isinstance(weight_buffers._managed_tensors, list):
                    raise RuntimeError(
                        "Invalid `weight_buffers._managed_tensors`: expected list "
                        f"for tensor lifetime tracking, got "
                        f"{type(weight_buffers._managed_tensors).__name__}"
                    )

                target_device = torch.device("mps")

                def convert_field(
                    attr_name: str,
                    tensor_attr: str,
                    expected_dtype: torch.dtype,
                    allowed_dtypes: tuple[torch.dtype, ...],
                    required: bool = True,
                ) -> None:
                    current_val = getattr(weight_buffers, attr_name, None)
                    # Already converted for C++ fast path
                    if (
                        current_val is not None
                        and type(current_val).__name__ == _MANAGED_BUFFER_TYPENAME
                    ):
                        return

                    tensor_val = getattr(weight_buffers, tensor_attr, None)
                    if tensor_val is None:
                        if required and current_val is None:
                            raise RuntimeError(
                                f"Missing required tensor input for C++ bridge: "
                                f"`weight_buffers.{attr_name}` is not a ManagedBuffer "
                                f"and `weight_buffers.{tensor_attr}` is absent."
                            )
                        if required and current_val is not None:
                            raise RuntimeError(
                                f"Unsupported cached buffer for field `{attr_name}`: "
                                f"`weight_buffers.{attr_name}` has type "
                                f"{type(current_val).__name__}, but C++ bridge requires "
                                f"a ManagedBuffer or a tensor mirror at "
                                f"`weight_buffers.{tensor_attr}` for normalization."
                            )
                        # Optional field not provided; skip conversion.
                        return

                    normalized = _normalize_tensor_for_cpp_bridge(
                        tensor_val,
                        field_name=attr_name,
                        tensor_attr=tensor_attr,
                        expected_device=target_device,
                        expected_dtype=expected_dtype,
                        allowed_dtypes=allowed_dtypes,
                    )

                    # Keep tensor alive for the ManagedBuffer lifetime.
                    clean_tensor = normalized.detach().clone(
                        memory_format=torch.contiguous_format
                    )
                    weight_buffers._managed_tensors.append(clean_tensor)

                    nbytes = clean_tensor.numel() * clean_tensor.element_size()
                    try:
                        managed = fast_path.create_buffer_from_ptr(
                            clean_tensor.data_ptr(),
                            nbytes,
                        )
                    except Exception as exc:  # pragma: no cover - ext/runtime dependent
                        raise RuntimeError(
                            f"Failed to wrap normalized tensor for field `{attr_name}` "
                            f"as ManagedBuffer (dtype={clean_tensor.dtype}, "
                            f"device={clean_tensor.device}, nbytes={nbytes}): {exc}"
                        ) from exc

                    setattr(weight_buffers, attr_name, managed)

                # Expert weights and scales
                convert_field(
                    "gate_weights",
                    "gate_weights_tensor",
                    torch.uint8,
                    _UINT8_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "gate_scales",
                    "gate_scales_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "up_weights",
                    "up_weights_tensor",
                    torch.uint8,
                    _UINT8_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "up_scales",
                    "up_scales_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "down_weights",
                    "down_weights_tensor",
                    torch.uint8,
                    _UINT8_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "down_scales",
                    "down_scales_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )

                # Sign vectors
                convert_field(
                    "gate_su",
                    "gate_su_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "gate_sv",
                    "gate_sv_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "up_su",
                    "up_su_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "up_sv",
                    "up_sv_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "down_su",
                    "down_su_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "down_sv",
                    "down_sv_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )

                # Grid and dynamic buffers
                convert_field(
                    "grid",
                    "grid_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                )
                convert_field(
                    "dynamic_grid_concat",
                    "_grid_concat_tensor",
                    torch.float16,
                    _HALF_COMPATIBLE_DTYPES,
                    required=False,
                )
                convert_field(
                    "dynamic_grid_offsets",
                    "_grid_offsets_tensor",
                    torch.int32,
                    _INT32_COMPATIBLE_DTYPES,
                    required=False,
                )

        # Import here to avoid circular dependency
        from .moe_dispatch import dispatch_moe_trellis_swiglu

        # For now, delegate to the standard dispatch with bit-grouped optimization
        # The dynamic dispatch groups experts by bit configuration and dispatches
        # each group with the appropriate kernel parameters

        output = torch.zeros(
            batch_size, hidden_dim,
            dtype=torch.float16,
            device=activations.device,
        )

        timing_enabled = os.getenv("METAL_MARLIN_DISPATCH_TIMING") == "1"
        timing_stats: dict[tuple[int, int, int], tuple[int, float]] = {}
        total_group_time = 0.0

        # Get or create device-specific cache for this dispatch
        device_key = str(expert_ids.device)
        if device_key not in self._group_expert_tensor_cache:
            self._group_expert_tensor_cache[device_key] = {}
        device_cache = self._group_expert_tensor_cache[device_key]

        # Process each unique bit group
        for bit_tuple, group_expert_indices in self._bit_groups.items():
            gate_bits, up_bits, down_bits = bit_tuple

            # Find which (batch, slot) positions use experts from this group
            # Use cached tensor if available, otherwise create and cache it
            if bit_tuple not in device_cache:
                device_cache[bit_tuple] = torch.tensor(
                    group_expert_indices, dtype=torch.long, device=expert_ids.device
                )
            group_expert_ids_t = device_cache[bit_tuple]
            mask = torch.isin(expert_ids, group_expert_ids_t)

            if not mask.any():
                continue  # No experts from this group selected

            batch_indices_t, slot_indices_t = torch.nonzero(mask, as_tuple=True)

            # Gather inputs for this bit group
            group_activations = activations[batch_indices_t]
            group_expert_ids = expert_ids[
                batch_indices_t, slot_indices_t
            ].unsqueeze(1)
            group_expert_probs = expert_probs[
                batch_indices_t, slot_indices_t
            ].unsqueeze(1).to(torch.float32)

            # Dispatch with this group's bit configuration
            # Use the maximum bit width for buffer sizing (conservative)
            max_bits = max(gate_bits, up_bits, down_bits)

            start_time = time.perf_counter() if timing_enabled else 0.0

            group_output = dispatch_moe_trellis_swiglu(
                lib=self._lib,
                activations=group_activations,
                gate_weights=gate_weights,
                gate_scales=gate_scales,
                up_weights=up_weights,
                up_scales=up_scales,
                down_weights=down_weights,
                down_scales=down_scales,
                gate_su=gate_su,
                gate_sv=gate_sv,
                up_su=up_su,
                up_sv=up_sv,
                down_su=down_su,
                down_sv=down_sv,
                grid=None,
                expert_ids=group_expert_ids,
                expert_probs=group_expert_probs,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=len(group_expert_indices),
                top_k=1,  # Each token only routes to one expert in this group
                bits=max_bits,  # Use max bits for this group
                cached_buffers=weight_buffers,
                buffer_pool=buffer_pool,
                use_fp32_acc=use_fp32_acc,
            )

            if timing_enabled:
                elapsed = time.perf_counter() - start_time
                total_group_time += elapsed
                prev_tokens, prev_time = timing_stats.get(bit_tuple, (0, 0.0))
                timing_stats[bit_tuple] = (
                    prev_tokens + batch_indices_t.shape[0],
                    prev_time + elapsed,
                )

            # Scatter results back to output
            output.index_add_(0, batch_indices_t, group_output.squeeze(1))

        if timing_enabled and timing_stats:
            header = "[dynamic_dispatch] group timing (tokens, ms, ms/token):"
            lines = [header]
            for bit_tuple, (token_count, elapsed) in sorted(timing_stats.items()):
                ms = elapsed * 1000.0
                ms_per = ms / max(token_count, 1)
                lines.append(
                    f"  bits={bit_tuple} tokens={token_count} "
                    f"time_ms={ms:.2f} ms_per_token={ms_per:.4f}"
                )
            lines.append(
                f"  total_group_time_ms={total_group_time * 1000.0:.2f}")
            print("\n".join(lines))

        return output

    def get_bit_groups(self) -> dict[tuple[int, int, int], list[int]]:
        """Get the bit group mapping.

        Returns:
            Dictionary mapping (gate_bits, up_bits, down_bits) tuples to
            lists of expert indices with that configuration.
        """
        return self._bit_groups.copy()

    @property
    def num_bit_groups(self) -> int:
        """Number of unique bit width groups."""
        return len(self._bit_groups)

    @property
    def num_experts(self) -> int:
        """Total number of experts."""
        return self._num_experts
