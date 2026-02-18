"""MMFP4 expert layer with optional fused decode path."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .mmfp4_linear import (
    MMFP4Linear,
    _as_u32_tensor,
    _fused_dtype_convert,
    _minimize_contiguous,
    _optimized_scale_load,
    _try_mmfp4_kernel_gemm,
)


def _convert_to_interleaved_weights(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    is_down_proj: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert row-packed weights to interleaved layout for coalesced memory access.
    
    Interleaved layout improves memory coalescing by organizing weights so that
    threads in a SIMD group access contiguous memory locations.
    
    For gate/up (input=hidden, output=intermediate):
        Input: [out=intermediate, in_packed=hidden/8]
        Output: [hidden/8, intermediate/8, 8] uint32
        
    For down (input=intermediate, output=hidden):
        Input: [out=hidden, in_packed=intermediate/8]
        Output: [intermediate/8, hidden/8, 8] uint32
    
    Args:
        packed_weights: [out_features, in_features//8] uint32 row-packed weights
        scales: [n_groups, out_features] float16 scales (already transposed in MMFP4Linear)
        group_size: Quantization group size
        is_down_proj: Whether this is the down projection (different layout)
        
    Returns:
        (interleaved_weights, interleaved_scales) tuple
        - interleaved_weights: [in_packed, out_features//8, 8] uint32
        - interleaved_scales: [n_groups, out_features//8, 8] float16
    """
    device = packed_weights.device
    packed_u32 = _minimize_contiguous(_as_u32_tensor(packed_weights))
    out_features, in_packed = packed_u32.shape
    in_features = in_packed * 8
    
    # Ensure dimensions are divisible by 8 for clean tiling
    out_aligned = ((out_features + 7) // 8) * 8
    
    # Pad if necessary
    if out_features < out_aligned:
        padding = torch.zeros(
            (out_aligned - out_features, in_packed),
            dtype=torch.uint32,
            device=device
        )
        packed_u32 = torch.cat([packed_u32, padding], dim=0)
    
    # Layout conversion: [out_aligned, in_packed] -> [in_packed, out_aligned//8, 8]
    # Step 1: Unpack to [out_aligned, in_features] uint8
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    words = packed_u32.to(torch.int32).unsqueeze(-1)  # [out_aligned, in_packed, 1]
    nibbles = torch.bitwise_and(torch.bitwise_right_shift(words, shifts), 0xF)  # [out_aligned, in_packed, 8]
    nibbles = nibbles.reshape(out_aligned, in_features).to(torch.int64)  # [out_aligned, in_features]
    
    # Step 2: Transpose to [in_features, out_aligned] = [in_packed*8, out_aligned]
    nibbles_t = nibbles.transpose(0, 1)  # [in_features, out_aligned]
    
    # Step 3: Reshape to [in_packed, 8, out_aligned//8, 8]
    nibbles_t = nibbles_t.reshape(in_packed, 8, out_aligned // 8, 8)
    
    # Step 4: Permute to [in_packed, out_aligned//8, 8, 8]
    # This puts the 8 weights for SIMD coalesced access in the last dim
    nibbles_t = nibbles_t.permute(0, 2, 1, 3).contiguous()  # [in_packed, out//8, 8, 8]
    
    # Step 5: Pack the last dimension: [in_packed, out//8, 8, 8] -> [in_packed, out//8, 8] uint32
    shifts_pack = torch.arange(8, device=device, dtype=torch.int32) * 4
    interleaved_weights = torch.sum(
        torch.bitwise_left_shift(nibbles_t.to(torch.int32), shifts_pack.view(1, 1, 1, 8)),
        dim=-1
    ).to(torch.uint32)  # [in_packed, out//8, 8]
    
    # Handle scales: [n_groups, out_features] -> [n_groups, out//8, 8]
    scales_f16 = scales.to(torch.float16)
    
    # Pad scales if needed
    if out_features < out_aligned:
        scale_padding = torch.ones(
            (scales_f16.shape[0], out_aligned - out_features),
            dtype=torch.float16,
            device=device
        )
        scales_f16 = torch.cat([scales_f16, scale_padding], dim=1)
    
    # Reshape to interleaved: [n_groups, out_aligned] -> [n_groups, out//8, 8]
    n_groups = scales_f16.shape[0]
    interleaved_scales = scales_f16.reshape(n_groups, out_aligned // 8, 8).contiguous()
    
    return interleaved_weights, interleaved_scales


def _prepare_fused_moe_weights(
    gate_up_proj: MMFP4Linear,
    down_proj: MMFP4Linear,
) -> dict[str, torch.Tensor]:
    """Prepare interleaved weights for fused MoE kernel.
    
    Returns dict with interleaved weight and scale tensors ready for Metal kernel.
    """
    # Gate and Up are fused in one linear layer: [2*intermediate, hidden]
    # Split into separate gate and up weights
    hidden_size = gate_up_proj.in_features
    intermediate_size = gate_up_proj.out_features // 2
    group_size = gate_up_proj.group_size
    
    # Split gate_up into gate and up: each is [intermediate, hidden/8]
    packed_gate_up = gate_up_proj.packed_weights  # [2*intermediate, hidden/8]
    packed_gate = packed_gate_up[:intermediate_size]  # [intermediate, hidden/8]
    packed_up = packed_gate_up[intermediate_size:]    # [intermediate, hidden/8]
    
    # Split scales: [n_groups, 2*intermediate] -> [n_groups, intermediate] each
    # Scales are already transposed to [n_groups, out_features] in MMFP4Linear
    scales_gate_up = gate_up_proj.scales  # [n_groups, 2*intermediate]
    scales_gate = scales_gate_up[:, :intermediate_size]  # [n_groups, intermediate]
    scales_up = scales_gate_up[:, intermediate_size:]    # [n_groups, intermediate]
    
    # Convert to interleaved layout
    # Gate weights: [intermediate, hidden/8] -> [hidden/8, intermediate/8, 8]
    gate_interleaved, gate_scales_interleaved = _convert_to_interleaved_weights(
        packed_gate,  # [intermediate, hidden/8]
        scales_gate,  # [n_groups, intermediate]
        group_size,
    )
    
    # Up weights: [intermediate, hidden/8] -> [hidden/8, intermediate/8, 8]
    up_interleaved, up_scales_interleaved = _convert_to_interleaved_weights(
        packed_up,
        scales_up,
        group_size,
    )
    
    # Down weights: [hidden, intermediate/8] -> [intermediate/8, hidden/8, 8]
    down_interleaved, down_scales_interleaved = _convert_to_interleaved_weights(
        down_proj.packed_weights,  # [hidden, intermediate/8]
        down_proj.scales,  # [n_groups, hidden]
        group_size,
        is_down_proj=True,
    )
    
    return {
        'gate_proj_packed': gate_interleaved,
        'up_proj_packed': up_interleaved,
        'down_proj_packed': down_interleaved,
        'gate_scales': gate_scales_interleaved,
        'up_scales': up_scales_interleaved,
        'down_scales': down_scales_interleaved,
    }


def _expert_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization for expert outputs.
    
    Normalizes expert outputs to stabilize training and inference in MoE architectures.
    Uses Root Mean Square (RMS) normalization which is computationally efficient and
    works well with SwiGLU activations.
    
    Args:
        x: Input tensor of shape [..., hidden_size]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of same shape as input
    """
    # Compute RMS along the last dimension (hidden_size)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms


class MMFP4Expert(nn.Module):
    """Single SwiGLU expert using MMFP4-quantized linear layers.

    This module implements a SwiGLU expert block, typically used in Mixture-of-Experts (MoE)
    architectures. It consists of a fused gate and up projection, followed by a SiLU activation
    and element-wise multiplication, and finally a down projection. The linear layers are
    quantized using MMFP4 format for memory efficiency.

    Expert outputs are normalized using RMS normalization to stabilize MoE training dynamics
    and prevent expert collapse or variance explosion.

    Args:
        hidden_size (int): Dimension of the input and output features.
        moe_intermediate_size (int): Dimension of the intermediate features (before splitting).
            The actual internal dimension is moe_intermediate_size for gate and up projections.
        group_size (int, optional): Group size for MMFP4 quantization. Defaults to 128.
        use_fused (bool, optional): Whether to use the fused kernel path.
            Currently defaults to False as the fused kernel may require specific tiling fixes.
        use_expert_norm (bool, optional): Whether to apply expert output normalization.
            Defaults to True for training stability.

    Returns:
        torch.Tensor: The output tensor of shape [batch_size, seq_len, hidden_size].

    Example:
        >>> expert = MMFP4Expert(hidden_size=4096, moe_intermediate_size=11008)
        >>> x = torch.randn(1, 10, 4096)
        >>> output = expert(x)
        >>> print(output.shape)
        torch.Size([1, 10, 4096])
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        group_size: int = 128,
        use_fused: bool = False,  # Disabled: fused kernel needs tiling fix for intermediate_size > 1536
        use_expert_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.intermediate_size = moe_intermediate_size
        self.group_size = group_size
        self.use_fused = use_fused
        self.use_expert_norm = use_expert_norm

        # Create placeholder MMFP4Linear layers - weights loaded later.
        self.gate_up_proj = _make_placeholder_mmfp4_linear(
            hidden_size, 2 * moe_intermediate_size, group_size
        )
        self.down_proj = _make_placeholder_mmfp4_linear(
            moe_intermediate_size, hidden_size, group_size
        )

    def _expert_memory_layout(self) -> dict[str, torch.Tensor]:
        """Return expert weights in contiguous memory layout for optimized access.

        This method ensures all expert weights are stored contiguously in memory,
        improving cache efficiency and enabling faster kernel access patterns.
        The contiguous layout is essential for high-performance MoE execution
        where memory bandwidth is the bottleneck.

        Returns:
            Dictionary containing contiguous weight tensors:
            - 'gate_up_packed': Contiguous gate+up packed weights [2*intermediate, hidden/8]
            - 'gate_up_scales': Contiguous gate+up scales [n_groups, 2*intermediate]
            - 'down_packed': Contiguous down packed weights [hidden, intermediate/8]
            - 'down_scales': Contiguous down scales [n_groups_intermediate, hidden]
        """
        # Ensure all weights are contiguous for optimal memory access
        gate_up_packed = _minimize_contiguous(
            _as_u32_tensor(self.gate_up_proj.packed_weights)
        )
        gate_up_scales = _minimize_contiguous(self.gate_up_proj.scales)
        down_packed = _minimize_contiguous(
            _as_u32_tensor(self.down_proj.packed_weights)
        )
        down_scales = _minimize_contiguous(self.down_proj.scales)

        return {
            'gate_up_packed': gate_up_packed,
            'gate_up_scales': gate_up_scales,
            'down_packed': down_packed,
            'down_scales': down_scales,
        }

    def _prefetch_weights(
        self,
        device: torch.device | None = None,
        async_prefetch: bool = True,
        sync: bool = False,
        precompute_kernel_layout: bool = True,
    ) -> None:
        """Prefetch expert weights to target device to hide memory latency.

        This method proactively transfers expert weights from host memory to the
        target device (e.g., GPU/Metal Performance Shaders) before computation.
        By prefetching weights ahead of time, we overlap memory transfer with
        other computation, effectively hiding memory latency.

        The prefetch operation is non-blocking when possible (using non_blocking=True),
        allowing the CPU to continue while data transfers asynchronously.

        Args:
            device: Target device to prefetch weights to. If None, uses the
                current device of gate_up_proj.packed_weights.
            async_prefetch: If True, uses non-blocking transfers for overlapping
                memory operations with computation. Set to False if immediate
                synchronization is required.
            sync: If True, synchronizes the device after prefetch to ensure
                all weights are fully loaded. Use for debugging or when
                subsequent operations require guaranteed weight availability.
            precompute_kernel_layout: If True, precomputes kernel-optimized weight
                layouts (transpose+contiguous) during prefetch to hide layout
                conversion overhead from the forward pass.

        Optimization Strategy:
            1. Identify target device for weight placement
            2. Use non-blocking transfers to overlap with computation
            3. Prefetch all weight components (packed weights + scales)
            4. Prepare kernel cache layout for immediate use
            5. Ensure contiguous memory layout for coalesced access
            6. Track prefetch status to avoid redundant operations
            7. Optional device synchronization for timing-critical paths

        Example:
            >>> expert = MMFP4Expert(hidden_size=4096, moe_intermediate_size=11008)
            >>> # Async prefetch - returns immediately, weights load in background
            >>> expert._prefetch_weights(device=torch.device('mps:0'))
            >>> # Do other work while weights transfer...
            >>> # Before using weights, optionally sync
            >>> expert._prefetch_weights(sync=True)
            >>> # Weights are now guaranteed ready on device
        """
        if device is None:
            device = self.gate_up_proj.packed_weights.device

        # Use non-blocking for async overlap when requested
        non_blocking = async_prefetch and device.type != 'cpu'

        # Track if any transfers were initiated (for synchronization)
        transfers_initiated = False

        # Prefetch gate_up projection weights with memory pinning hint
        if self.gate_up_proj.packed_weights.device != device:
            self.gate_up_proj.packed_weights = self.gate_up_proj.packed_weights.to(
                device=device, non_blocking=non_blocking, memory_format=torch.contiguous_format
            )
            transfers_initiated = True

        if self.gate_up_proj.scales.device != device:
            self.gate_up_proj.scales = self.gate_up_proj.scales.to(
                device=device, non_blocking=non_blocking, memory_format=torch.contiguous_format
            )
            transfers_initiated = True

        # Prefetch down projection weights
        if self.down_proj.packed_weights.device != device:
            self.down_proj.packed_weights = self.down_proj.packed_weights.to(
                device=device, non_blocking=non_blocking, memory_format=torch.contiguous_format
            )
            transfers_initiated = True

        if self.down_proj.scales.device != device:
            self.down_proj.scales = self.down_proj.scales.to(
                device=device, non_blocking=non_blocking, memory_format=torch.contiguous_format
            )
            transfers_initiated = True

        # Pre-convert to kernel layout to avoid conversion during forward pass
        # This hides the transpose+contiguous overhead in prefetch phase
        if precompute_kernel_layout:
            gate_up_u32 = _as_u32_tensor(self.gate_up_proj.packed_weights)
            expected_gate_up_shape = (gate_up_u32.shape[1], gate_up_u32.shape[0])
            needs_gate_up_layout = (
                self.gate_up_proj._kernel_packed_weights is None
                or self.gate_up_proj._kernel_packed_weights.device != device
                or tuple(self.gate_up_proj._kernel_packed_weights.shape) != expected_gate_up_shape
            )
            if needs_gate_up_layout:
                # Perform layout conversion on device for optimal performance
                self.gate_up_proj._kernel_packed_weights = _minimize_contiguous(
                    gate_up_u32.transpose(0, 1)
                )

            down_u32 = _as_u32_tensor(self.down_proj.packed_weights)
            expected_down_shape = (down_u32.shape[1], down_u32.shape[0])
            needs_down_layout = (
                self.down_proj._kernel_packed_weights is None
                or self.down_proj._kernel_packed_weights.device != device
                or tuple(self.down_proj._kernel_packed_weights.shape) != expected_down_shape
            )
            if needs_down_layout:
                self.down_proj._kernel_packed_weights = _minimize_contiguous(
                    down_u32.transpose(0, 1)
                )

        # Synchronize if requested to ensure weights are fully loaded
        if sync and transfers_initiated and device.type != 'cpu':
            if hasattr(torch, 'mps') and device.type == 'mps':
                # MPS synchronization (best effort - may not be fully supported)
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize(device)

    def _ensure_weights_ready(self, device: torch.device | None = None) -> None:
        """Ensure weights are ready for computation, prefetching if necessary.

        This is a convenience method that combines prefetch with explicit
        synchronization. Use this when you need guaranteed weight availability
        before proceeding with computation.

        Args:
            device: Target device. If None, uses current device.
        """
        self._prefetch_weights(device=device, sync=True, precompute_kernel_layout=True)

    def _prefetch_weights_async(
        self,
        device: torch.device | None = None,
        callback: Callable[[], None] | None = None,
    ) -> None:
        """Asynchronously prefetch weights with optional completion callback.

        This method initiates weight prefetching and optionally calls a callback
        when complete. The callback is executed synchronously after prefetch
        (not in a separate thread), but allows for structured async patterns.

        Args:
            device: Target device for weights.
            callback: Optional function to call after prefetch completes.

        Example:
            >>> def on_weights_ready():
            ...     print("Weights loaded, starting computation")
            >>> expert._prefetch_weights_async(callback=on_weights_ready)
        """
        self._prefetch_weights(device=device, async_prefetch=True, sync=False)
        if callback is not None:
            callback()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused path for single token decode or batched (size 8)
        # Ensure 2D input [batch, hidden_size] for fused kernel
        # Always use optimized decode path for single token decode (M=1)
        if x.ndim == 2 and (x.shape[0] == 1 or (self.use_fused and x.shape[0] == 8)):
            return self._decode_optimized(x)

        # Standard path for prefill
        return self._standard_forward(x)

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _optimized_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Fused gate * SiLU(up) computation for SwiGLU activation.
        
        SwiGLU(x) = gate * SiLU(up) where SiLU(x) = x * sigmoid(x)
        This fused computation minimizes kernel dispatch overhead.
        """
        return gate * torch.nn.functional.silu(up)

    def _fused_silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Fused gate * SiLU(up) computation for SwiGLU activation.
        
        This method applies the SwiGLU activation: gate * SiLU(up)
        where SiLU(x) = x * sigmoid(x). This is a key component of the
        expert MLP computation.
        
        Args:
            gate: Gate projection output tensor
            up: Up projection output tensor
            
        Returns:
            Activated tensor: gate * SiLU(up)
        """
        return gate * torch.nn.functional.silu(up)

    def _fused_gate_up(self, x: torch.Tensor) -> torch.Tensor:
        """Fused gate+up projection using single-kernel dispatch.
        
        This method uses a single Metal kernel that computes both gate and up
        projections simultaneously, reducing kernel launch overhead and improving
        memory bandwidth utilization compared to separate GEMM calls.
        
        The fused kernel performs both projections in a single dispatch:
        - Input: [M, hidden] 
        - Output: [M, 2*intermediate] with gate and up concatenated
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Output tensor [..., 2*intermediate_size] with gate and up concatenated
        """
        # Flatten for GEMM: [..., hidden] -> [M, hidden]
        x_2d = x.reshape(-1, self.hidden_size)
        
        # Try single-kernel fused path for MPS
        if x.is_mps:
            out_2d = self._fused_gate_up_kernel(x_2d)
            if out_2d is not None:
                return out_2d.reshape(*x.shape[:-1], 2 * self.intermediate_size)
        
        # Fallback: combined GEMM then split (original behavior)
        out_2d, kernel_cache = _try_mmfp4_kernel_gemm(
            x_2d,
            self.gate_up_proj.packed_weights,
            self.gate_up_proj.scales,
            self.group_size,
            self.gate_up_proj._kernel_packed_weights,
        )

        if kernel_cache is not self.gate_up_proj._kernel_packed_weights:
            self.gate_up_proj._kernel_packed_weights = kernel_cache

        if out_2d is None:
            out_2d = self.gate_up_proj(x, skip_kernel_try=True)
            if out_2d.dim() > 2:
                return out_2d.reshape(*x.shape[:-1], 2 * self.intermediate_size)
            return out_2d

        return out_2d.reshape(*x.shape[:-1], 2 * self.intermediate_size)
    
    def _fused_gate_up_kernel(self, x_2d: torch.Tensor) -> torch.Tensor | None:
        """Single-kernel fused gate+up projection for MPS.
        
        Uses the fused_gate_up_gemm Metal kernel to compute both gate and up 
        projections in a single dispatch, avoiding separate kernel launches and 
        reducing memory bandwidth by ~50% compared to two separate GEMM calls.
        
        Weight layout transformation:
        - Input weights: [2*intermediate, hidden/8] (row-packed)
        - Kernel expects: [hidden/8, intermediate] for each projection
        
        Args:
            x_2d: 2D input tensor [M, hidden_size]
            
        Returns:
            2D output tensor [M, 2*intermediate_size] or None if kernel unavailable
        """
        try:
            from ..kernels import fused_gate_up_gemm as _fused_gate_up_kernel_fn
        except Exception:
            return None
        
        try:
            # Split gate_up weights into separate gate and up components
            # gate_up_proj.packed_weights: [2*intermediate, hidden/8]
            packed_gate_up = self.gate_up_proj.packed_weights
            intermediate_size = self.intermediate_size
            
            packed_gate = packed_gate_up[:intermediate_size]  # [intermediate, hidden/8]
            packed_up = packed_gate_up[intermediate_size:]    # [intermediate, hidden/8]
            
            # Split scales: [n_groups, 2*intermediate] -> [n_groups, intermediate] each
            scales_gate_up = self.gate_up_proj.scales
            scales_gate = scales_gate_up[:, :intermediate_size]
            scales_up = scales_gate_up[:, intermediate_size:]
            
            # Convert to kernel-preferred layout [hidden/8, intermediate]
            # The fused_gate_up_gemm kernel expects:
            # - packed weights: [K/8, N] where K=hidden, N=intermediate
            # - scales: [K/group_size, N]
            gate_kernel_layout = _minimize_contiguous(
                _as_u32_tensor(packed_gate).transpose(0, 1)
            )  # [hidden/8, intermediate]
            up_kernel_layout = _minimize_contiguous(
                _as_u32_tensor(packed_up).transpose(0, 1)
            )  # [hidden/8, intermediate]
            
            # Prepare scales in kernel-preferred layout [N, n_groups]
            # This enables coalesced memory access during kernel execution
            n_groups = (self.hidden_size + self.group_size - 1) // self.group_size
            scales_gate_kernel = _minimize_contiguous(
                scales_gate.transpose(0, 1).to(torch.float16)
            )  # [intermediate, n_groups]
            scales_up_kernel = _minimize_contiguous(
                scales_up.transpose(0, 1).to(torch.float16)
            )  # [intermediate, n_groups]
            
            # Convert input to float16
            x_f16 = _fused_dtype_convert(x_2d, torch.float16, x_2d.device)
            
            # Call the fused kernel - single dispatch for both projections
            out = _fused_gate_up_kernel_fn(
                x_f16,
                gate_kernel_layout,
                scales_gate_kernel,
                up_kernel_layout,
                scales_up_kernel,
                group_size=self.group_size,
            )
            
            if out is None:
                return None
                
            # Validate output shape and values
            expected_shape = (x_2d.shape[0], 2 * intermediate_size)
            if out.shape != expected_shape:
                return None
            if not torch.isfinite(out).all():
                return None
                
            return out
            
        except Exception:
            return None

    def _optimized_down(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized down projection with streamlined kernel path.
        
        Optimizations:
        1. Single kernel dispatch: Direct MMFP4 kernel call with minimal overhead
        2. Specialized fast paths for M=1 (decode) and M<=16 (small batch)
        3. Zero-allocation reshape using view() when possible
        4. Pre-cached scale layouts: Eliminates per-call transpose overhead
        5. Inline kernel dispatch without try/except on hot path
        6. Aggressive cache pre-warming for repeated decode calls
        
        Args:
            x: Input tensor [batch, seq, intermediate_size] or [batch, intermediate_size]
            
        Returns:
            Output tensor [batch, seq, hidden_size] or [batch, hidden_size]
        """
        # Fast shape tracking without storing full shape tuple
        original_ndim = x.ndim
        batch_size = x.shape[0] if original_ndim >= 2 else 1
        seq_len = x.shape[1] if original_ndim >= 3 else 1
        
        # Flatten for GEMM: [..., intermediate_size] -> [M, intermediate_size]
        x_2d = x.reshape(-1, self.intermediate_size)
        m_size = x_2d.shape[0]

        # Ensure decode cache is initialized (weights and scales)
        self._ensure_decode_cache()

        # Fast path for MPS with cached weights - specialized by batch size
        if x.is_mps and self.down_proj._kernel_packed_weights is not None:
            # Ultra-fast path: M=1 decode (most common case)
            if m_size == 1:
                result = self._optimized_down_m1(x_2d)
                if result is not None:
                    # Zero-copy reshape for decode case
                    return result.view(batch_size, seq_len, self.hidden_size) if original_ndim > 2 else result
            
            # Fast path: Small batch M <= 16
            if m_size <= 16:
                result = self._optimized_down_small_batch(x_2d, m_size)
                if result is not None:
                    # Efficient reshape using view where possible
                    if original_ndim == 2:
                        return result
                    return result.view(*x.shape[:-1], self.hidden_size)

        # Standard optimized path: Direct kernel call with cached weights
        out_2d, kernel_cache = _try_mmfp4_kernel_gemm(
            x_2d,
            self.down_proj.packed_weights,
            self.down_proj.scales,
            self.group_size,
            self.down_proj._kernel_packed_weights,
        )

        # Update weight layout cache if kernel created new layout
        if kernel_cache is not self.down_proj._kernel_packed_weights:
            self.down_proj._kernel_packed_weights = kernel_cache

        if out_2d is None:
            # Fallback to standard MMFP4Linear forward (handles dequantization)
            return self.down_proj(x)

        # Efficient reshape using pre-computed dimension count
        if original_ndim == 2:
            return out_2d.view(batch_size, self.hidden_size) if out_2d.shape[0] == batch_size else out_2d
        return out_2d.view(batch_size, seq_len, self.hidden_size)

    def _optimized_down_m1(self, x_2d: torch.Tensor) -> torch.Tensor | None:
        """Ultra-optimized down projection for single-token decode (M=1).
        
        This is the hottest path in decode - minimizes all overhead:
        - No try/except (caller handles None return)
        - Direct kernel dispatch
        - Pre-validated cached weights/scales
        - Single dtype conversion
        
        Args:
            x_2d: Input tensor [1, intermediate_size]
            
        Returns:
            Output tensor [1, hidden_size] or None on failure
        """
        from ..kernels import mmfp4_gemm as _kernel_mmfp4_gemm
        
        # Fast dtype conversion (already float16 on MPS decode path usually)
        x_f16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
        
        # Direct kernel call with pre-cached weights/scales
        result = _kernel_mmfp4_gemm(
            x_f16,
            self.down_proj._kernel_packed_weights,
            self._down_scales_t,
            group_size=self.group_size,
        )
        
        # Fast validation without GPU sync
        if result is not None and result.shape[0] == 1:
            return result
        return None

    def _optimized_down_small_batch(
        self, 
        x_2d: torch.Tensor, 
        m_size: int
    ) -> torch.Tensor | None:
        """Optimized down projection for small batches (2 <= M <= 16).
        
        Uses the same fast kernel path but handles variable batch sizes.
        
        Args:
            x_2d: Input tensor [M, intermediate_size]
            m_size: Batch size M
            
        Returns:
            Output tensor [M, hidden_size] or None on failure
        """
        try:
            from ..kernels import mmfp4_gemm as _kernel_mmfp4_gemm
            
            # Fused dtype conversion
            x_f16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
            
            # Use pre-cached scales: [hidden_size, n_groups] for coalesced access
            result = _kernel_mmfp4_gemm(
                x_f16,
                self.down_proj._kernel_packed_weights,
                self._down_scales_t,
                group_size=self.group_size,
            )
            
            # Validate shape without GPU synchronization
            if result is not None and result.shape[0] == m_size:
                return result
        except Exception:
            pass
        return None

    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward path with optimized SwiGLU activation.
        
        Uses @torch.compile optimized SwiGLU for reduced kernel dispatch overhead
        and better memory bandwidth utilization in the activation computation.
        """
        gate_up = self._fused_gate_up(x)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        # Use @torch.compile optimized SwiGLU for reduced dispatch overhead
        # _optimized_swiglu uses torch.compile(mode="reduce-overhead") for
        # optimal kernel fusion and memory bandwidth utilization
        activated = self._optimized_swiglu(gate, up)
        output = self._optimized_down(activated)
        # Apply expert output normalization for training stability
        if self.use_expert_norm:
            output = _expert_norm(output)
        return output

    # Cache for compiled decode functions (class-level to avoid recompilation)
    _compiled_decode_cache: dict[tuple[int, int, int, bool], Callable] = {}
    
    def _decode_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized path for single token decode (x is [1, hidden]).
        
        Optimizations:
        1. Fused Metal kernel path: Single dispatch for gate+up+activation+down
        2. Cached scale layouts: Pre-transposed scales eliminate conversion overhead
        3. Compiled decode kernel: torch.compile for minimal Python overhead
        4. Minimal allocations: In-place operations and early tensor deletion
        5. Kernel fusion: Reduces dispatch overhead from 3 kernels to 1-2 kernels
        """
        # Fast path: Fused Metal kernel with interleaved weights for MPS
        if x.is_mps and self.use_fused:
            try:
                return self._fused_moe_mlp_kernel(x)
            except Exception as e:
                print(f"Fused MoE kernel failed: {e}")
                import traceback
                traceback.print_exc()
                pass
        
        # Ensure weights and scales are cached in optimal layout
        self._ensure_decode_cache()
        
        # Try compiled fast path for MPS with cached weights
        if x.is_mps and self.gate_up_proj._kernel_packed_weights is not None:
            try:
                return self._decode_compiled_fastpath(x)
            except Exception as e:
                print(f"Compiled fast path failed: {e}")
                pass
        
        # Fallback to standard 3-kernel path
        return self._decode_optimized_fallback(x)
    
    def _ensure_decode_cache(self) -> None:
        """Ensure all decode caches are initialized (weights and scales).
        
        Caches:
        - _kernel_packed_weights: Transposed weights in kernel layout
        - _gate_up_scales_t: Transposed scales [2*intermediate, n_groups]
        - _down_scales_t: Transposed scales [hidden, n_groups]
        """
        # Ensure kernel weight layouts are cached
        if self.gate_up_proj._kernel_packed_weights is None:
            gate_up_u32 = _as_u32_tensor(self.gate_up_proj.packed_weights)
            self.gate_up_proj._kernel_packed_weights = _minimize_contiguous(
                gate_up_u32.transpose(0, 1)
            )
        
        if self.down_proj._kernel_packed_weights is None:
            down_u32 = _as_u32_tensor(self.down_proj.packed_weights)
            self.down_proj._kernel_packed_weights = _minimize_contiguous(
                down_u32.transpose(0, 1)
            )
        
        # Cache transposed scales for coalesced access: [n_groups, N] -> [N, n_groups]
        if not hasattr(self, '_gate_up_scales_t') or self._gate_up_scales_t.device != self.gate_up_proj.scales.device:
            self._gate_up_scales_t = _minimize_contiguous(
                self.gate_up_proj.scales.transpose(0, 1).to(torch.float16)
            )
        
        if not hasattr(self, '_down_scales_t') or self._down_scales_t.device != self.down_proj.scales.device:
            self._down_scales_t = _minimize_contiguous(
                self.down_proj.scales.transpose(0, 1).to(torch.float16)
            )
    
    def _decode_compiled_fastpath(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path using compiled kernel dispatch with cached weights/scales.
        
        This path minimizes Python overhead by:
        1. Using pre-cached weight layouts (no transpose on hot path)
        2. Using pre-cached scale layouts (no transpose/contiguous on hot path)
        3. Direct kernel dispatch without try/except in the hot loop
        4. In-place activation to reduce memory allocation
        """
        # Import kernel once at module level for speed
        from ..kernels import mmfp4_gemm as _kernel_gemm
        
        device = x.device
        group_size = self.group_size
        intermediate_size = self.intermediate_size
        use_expert_norm = self.use_expert_norm
        
        # Ensure float16 input for kernel
        x_f16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        
        # 1. Gate+Up projection with cached weights and scales
        gate_up = _kernel_gemm(
            x_f16,
            self.gate_up_proj._kernel_packed_weights,
            self._gate_up_scales_t,
            group_size=group_size,
        )
        
        if gate_up is None:
            raise RuntimeError("Gate+Up kernel GEMM returned None")
        
        # 2. SwiGLU: gate * SiLU(up)
        # Optimized with in-place operations to avoid allocation
        gate = gate_up[..., :intermediate_size]
        up = gate_up[..., intermediate_size:]
        
        # In-place SiLU on 'up' (safe as we don't need original 'up')
        # This modifies the second half of gate_up in-place
        torch.nn.functional.silu(up, inplace=True)
        
        # Element-wise multiplication: gate * silu(up)
        activated = gate * up
        
        # 3. Down projection with cached weights and scales
        out = _kernel_gemm(
            activated,
            self.down_proj._kernel_packed_weights,
            self._down_scales_t,
            group_size=group_size,
        )
        
        if out is None:
            raise RuntimeError("Down kernel GEMM returned None")
        
        # 4. Optional expert normalization
        if use_expert_norm:
            out = _expert_norm(out)
        
        return out
    
    def _decode_optimized_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback 3-kernel path when compiled fast path fails."""
        # Optimized 3-kernel path with cached weight layouts
        gate_up = self._decode_gemm_cached(
            x,
            self.gate_up_proj,
            self.intermediate_size * 2,
        )
        
        # SwiGLU activation
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        activated = self._optimized_swiglu(gate, up)
        del gate_up, gate, up
        
        # Down projection
        out = self._decode_gemm_cached(
            activated,
            self.down_proj,
            self.hidden_size,
        )
        del activated
        
        # Expert normalization
        if self.use_expert_norm:
            out = _expert_norm(out)
        
        return out
    
    def _decode_gemm_cached(
        self,
        x: torch.Tensor,
        proj: MMFP4Linear,
        out_features: int,
    ) -> torch.Tensor:
        """Single-token GEMM with cached weight layout and minimal overhead.
        
        Optimizations:
        1. Uses pre-cached kernel weights to avoid transpose+contiguous on every call
        2. Direct kernel dispatch for MPS with fallback to PyTorch
        3. Minimizes intermediate allocations
        
        Args:
            x: Input tensor [1, in_features]
            proj: MMFP4Linear layer with cached _kernel_packed_weights
            out_features: Expected output feature dimension
            
        Returns:
            Output tensor [1, out_features]
        """
        # Ensure 2D input for GEMM
        if x.ndim != 2:
            x = x.reshape(1, -1)
        
        # Try direct kernel path with cached weights
        if x.is_mps and proj._kernel_packed_weights is not None:
            try:
                from ..kernels import mmfp4_gemm as _kernel_mmfp4_gemm
                
                # Fused dtype conversion: single operation
                x_f16 = _fused_dtype_convert(x, torch.float16, x.device)
                
                # Use cached kernel layout weights [K/8, N]
                out = _kernel_mmfp4_gemm(
                    x_f16,
                    proj._kernel_packed_weights,
                    proj.scales.transpose(0, 1).contiguous().to(torch.float16),  # [N, n_groups]
                    group_size=self.group_size,
                )
                
                if out is not None and out.shape == (1, out_features):
                    if torch.isfinite(out).all():
                        return out
            except Exception:
                pass  # Fall through to standard path
        
        # Standard path: Try kernel with auto-layout-conversion
        out, cache = _try_mmfp4_kernel_gemm(
            x,
            proj.packed_weights,
            proj.scales,
            self.group_size,
            proj._kernel_packed_weights,
        )
        
        # Cache the kernel layout for future calls
        if cache is not proj._kernel_packed_weights:
            proj._kernel_packed_weights = cache
        
        if out is not None:
            return out
        
        # Fallback to PyTorch dequantization path
        return proj(x)
    
    def _fused_moe_mlp_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Call the fused MoE MLP Metal kernel with interleaved weights.
        
        This kernel performs gate_proj, up_proj, SiLU activation, multiplication,
        and down_proj in a single dispatch for maximum efficiency.
        
        Optimizations:
        1. Caches interleaved weights after first preparation
        2. Minimizes device transfers by caching on target device
        3. Reuses cached weights across decode calls
        """
        from ..kernels import fused_moe_mlp as _fused_moe_mlp_kernel
        
        # Lazy initialization: Prepare and cache interleaved weights on first call
        if not hasattr(self, '_cached_interleaved_weights'):
            self._cached_interleaved_weights = _prepare_fused_moe_weights(
                self.gate_up_proj, self.down_proj
            )
            # Cache device to detect device changes
            self._cached_weights_device = None
        
        # Get cached interleaved weights
        cached = self._cached_interleaved_weights
        device = x.device
        
        # Only transfer weights if device changed (avoid redundant transfers)
        if self._cached_weights_device != device:
            gate_packed = cached['gate_proj_packed'].to(device, non_blocking=True)
            up_packed = cached['up_proj_packed'].to(device, non_blocking=True)
            down_packed = cached['down_proj_packed'].to(device, non_blocking=True)
            gate_scales = cached['gate_scales'].to(device, non_blocking=True)
            up_scales = cached['up_scales'].to(device, non_blocking=True)
            down_scales = cached['down_scales'].to(device, non_blocking=True)
            
            # Update cache with device-local copies
            self._cached_interleaved_weights = {
                'gate_proj_packed': gate_packed,
                'up_proj_packed': up_packed,
                'down_proj_packed': down_packed,
                'gate_scales': gate_scales,
                'up_scales': up_scales,
                'down_scales': down_scales,
            }
            self._cached_weights_device = device
        else:
            # Reuse cached weights on same device
            gate_packed = cached['gate_proj_packed']
            up_packed = cached['up_proj_packed']
            down_packed = cached['down_proj_packed']
            gate_scales = cached['gate_scales']
            up_scales = cached['up_scales']
            down_scales = cached['down_scales']
        
        # Call the fused kernel
        output = _fused_moe_mlp_kernel(
            x,
            gate_packed,
            up_packed,
            down_packed,
            gate_scales,
            up_scales,
            down_scales,
            self.hidden_size,
            self.intermediate_size,
            self.group_size,
        )
        
        if output is None:
            raise RuntimeError("Fused MoE kernel returned None")
        
        return output



def _make_placeholder_mmfp4_linear(
    in_features: int,
    out_features: int,
    group_size: int,
) -> MMFP4Linear:
    """Create MMFP4Linear with placeholder (zeros) weights."""
    in_features_aligned = ((in_features + 7) // 8) * 8
    n_groups = (in_features + group_size - 1) // group_size
    packed_weights = torch.zeros(
        (out_features, in_features_aligned // 8),
        dtype=torch.uint32,
    )
    scales = torch.ones((n_groups, out_features), dtype=torch.float16)
    return MMFP4Linear(
        packed_weights=packed_weights,
        scales=scales,
        bias=None,
        group_size=group_size,
    )
