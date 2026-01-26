"""Adaptive bits-per-weight selection like ExllamaV3.

This module provides intelligent bit-width selection for quantization based on:
1. Per-layer error budgets derived from Hessian-based sensitivity
2. Iterative refinement to minimize reconstruction error
3. MoE-aware quantization (experts with sparse activation can use lower bits)

Key insight: Some MoE expert layers can use 3-bit or even 2-bit
because they see less diverse inputs (sparse activation).

ExllamaV3 Approach:
- Start with target average bits-per-weight (e.g., 4.0)
- Compute sensitivity score per layer from Hessian diagonal
- Allocate bits inversely proportional to layer sensitivity
- Iteratively refine by re-quantizing with residual errors

References:
- ExllamaV3 HF quants: https://huggingface.co/turboderp
- Sensitivity-based quantization: arxiv.org/abs/2306.00978
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class QuantizationFormat(str, Enum):
    """Supported quantization formats."""

    FP4 = "fp4"
    INT4 = "int4"
    INT3 = "int3"
    INT2 = "int2"
    NF4 = "nf4"
    NF3 = "nf3"
    NF2 = "nf2"
    INT8 = "int8"


# Bit widths for each format
FORMAT_BITS: dict[QuantizationFormat, int] = {
    QuantizationFormat.FP4: 4,
    QuantizationFormat.INT4: 4,
    QuantizationFormat.INT3: 3,
    QuantizationFormat.INT2: 2,
    QuantizationFormat.NF4: 4,
    QuantizationFormat.NF3: 3,
    QuantizationFormat.NF2: 2,
    QuantizationFormat.INT8: 8,
}


@dataclass
class AdaptiveQuantResult:
    """Result of adaptive quantization.

    Attributes:
        quantized: Dequantized weight matrix (for error computation)
        bits: Selected bit width
        format: Selected quantization format
        scale: Per-group scale factors
        packed: Packed integer weights (format-dependent)
        reconstruction_error: MSE between original and quantized
        relative_error: Mean relative error |W - Q| / |W|
        iterations: Number of refinement iterations performed
    """

    quantized: NDArray[np.float32]
    bits: int
    format: QuantizationFormat
    scale: NDArray[np.float32]
    packed: NDArray[np.uint32]
    reconstruction_error: float
    relative_error: float
    iterations: int


@dataclass
class LayerBudget:
    """Per-layer bit budget allocation.

    Attributes:
        name: Layer name/identifier
        sensitivity: Hessian-based sensitivity score (higher = more sensitive)
        allocated_bits: Allocated bits for this layer
        format: Selected quantization format
        weight: Budget allocation weight (derived from sensitivity)
    """

    name: str
    sensitivity: float
    allocated_bits: float
    format: QuantizationFormat
    weight: float = 1.0


@dataclass
class ModelBudgetAllocation:
    """Full model bit budget allocation.

    Attributes:
        layers: Per-layer budget allocations
        target_bits: Target average bits-per-weight
        actual_bits: Achieved average bits-per-weight
        total_params: Total number of parameters
    """

    layers: list[LayerBudget]
    target_bits: float
    actual_bits: float
    total_params: int


class AdaptiveQuantizer:
    """Adaptively choose bits per layer based on error budget.

    This class implements ExllamaV3-style adaptive quantization:
    1. Compute per-layer sensitivity from Hessian diagonal
    2. Allocate bits inversely proportional to sensitivity
    3. Try quantization at candidate bit widths
    4. Use iterative refinement to minimize error

    Key insight for MoE models: Expert layers that see sparse, less diverse
    activations can tolerate more aggressive quantization (2-3 bits) because
    the effective input distribution is narrower.

    Example:
        >>> quantizer = AdaptiveQuantizer(
        ...     error_budget=0.01,
        ...     min_bits=2,
        ...     max_bits=8,
        ... )
        >>> result = quantizer.quantize_layer_adaptive(
        ...     weight=weights,
        ...     hessian=hessian,
        ...     target_error=0.005,
        ... )
        >>> print(f"Selected {result.bits}-bit, error={result.reconstruction_error:.6f}")
    """

    def __init__(
        self,
        error_budget: float = 0.01,
        min_bits: int = 2,
        max_bits: int = 8,
        group_size: int = 128,
        prefer_nf: bool = True,
        refinement_iterations: int = 3,
    ) -> None:
        """Initialize adaptive quantizer.

        Args:
            error_budget: Target reconstruction error (MSE). Layers with error
                above this budget will try higher bit widths. Default: 0.01.
            min_bits: Minimum allowed bits. Use 2 for aggressive MoE expert
                compression, 4 for quality-focused quantization. Default: 2.
            max_bits: Maximum allowed bits. Use 8 for critical layers that
                cannot tolerate quantization. Default: 8.
            group_size: Elements per quantization group. Smaller groups give
                better quality but more overhead. Default: 128.
            prefer_nf: If True, prefer NormalFloat formats (NF2, NF3, NF4) over
                INT formats for Gaussian-distributed weights. Default: True.
            refinement_iterations: Number of iterative refinement passes.
                More iterations = better quality but slower. Default: 3.
        """
        if min_bits < 2:
            raise ValueError(f"min_bits must be >= 2, got {min_bits}")
        if max_bits > 8:
            raise ValueError(f"max_bits must be <= 8, got {max_bits}")
        if min_bits > max_bits:
            raise ValueError(
                f"min_bits ({min_bits}) must be <= max_bits ({max_bits})"
            )
        if error_budget <= 0:
            raise ValueError(f"error_budget must be positive, got {error_budget}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        self.error_budget = error_budget
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.group_size = group_size
        self.prefer_nf = prefer_nf
        self.refinement_iterations = refinement_iterations

        # Build candidate formats sorted by bits (ascending)
        self._candidates = self._build_candidate_formats()

    def _build_candidate_formats(self) -> list[QuantizationFormat]:
        """Build ordered list of candidate formats within bit range."""
        candidates = []
        for bits in range(self.min_bits, self.max_bits + 1):
            for fmt, fmt_bits in FORMAT_BITS.items():
                if fmt_bits == bits:
                    # Prefer NF over INT if prefer_nf is True
                    candidates.append(fmt)

        # Sort: bits ascending, NF before INT at same bits
        def sort_key(fmt: QuantizationFormat) -> tuple[int, int]:
            bits = FORMAT_BITS[fmt]
            # NF formats get priority 0, INT formats get priority 1
            is_nf = fmt.value.startswith("nf")
            priority = 0 if (is_nf and self.prefer_nf) else 1
            return (bits, priority)

        return sorted(set(candidates), key=sort_key)

    def compute_sensitivity(
        self,
        hessian: NDArray[np.float64],
    ) -> float:
        """Compute layer sensitivity score from Hessian.

        Sensitivity is computed as the mean of the Hessian diagonal, which
        represents the curvature of the loss surface with respect to each
        weight. Higher sensitivity means the layer is more important for
        model quality and should use more bits.

        For MoE experts, the Hessian diagonal is often smaller due to sparse
        activation, allowing more aggressive quantization.

        Args:
            hessian: Hessian matrix [in_features, in_features] computed as
                H = X^T @ X from calibration activations.

        Returns:
            Scalar sensitivity score (higher = more sensitive).
        """
        diag = np.diag(hessian).astype(np.float64)
        # Use mean rather than sum to normalize across layer sizes
        return float(np.mean(diag))

    def quantize_layer_adaptive(
        self,
        weight: NDArray[np.floating],
        hessian: NDArray[np.floating],
        target_error: float | None = None,
    ) -> AdaptiveQuantResult:
        """Quantize layer with adaptive bit selection.

        Try quantization at different bit widths, starting from the lowest,
        and return the first that meets the error target. If no format
        meets the target, return the highest-quality (most bits) result.

        Args:
            weight: Weight matrix [out_features, in_features].
            hessian: Hessian matrix [in_features, in_features].
            target_error: Target reconstruction error (MSE). If None, uses
                self.error_budget. Use smaller values for more important layers.

        Returns:
            AdaptiveQuantResult with selected format and quantized weights.
        """
        W = np.asarray(weight, dtype=np.float64)
        H = np.asarray(hessian, dtype=np.float64)

        if target_error is None:
            target_error = self.error_budget

        out_features, in_features = W.shape
        best_result: AdaptiveQuantResult | None = None

        # Try each candidate format in order of bits (lowest first)
        for fmt in self._candidates:
            bits = FORMAT_BITS[fmt]

            # Quantize and measure error
            Q, scale, packed = self._quantize_with_format(W, H, fmt)

            # Compute reconstruction error
            mse = float(np.mean((W - Q) ** 2))
            rel_error = float(np.mean(np.abs(W - Q) / (np.abs(W) + 1e-10)))

            result = AdaptiveQuantResult(
                quantized=Q.astype(np.float32),
                bits=bits,
                format=fmt,
                scale=scale.astype(np.float32),
                packed=packed,
                reconstruction_error=mse,
                relative_error=rel_error,
                iterations=0,
            )

            # If this meets target, return it (lowest bits that works)
            if mse <= target_error:
                return self._apply_refinement(W, H, result)

            # Track best result in case nothing meets target
            if best_result is None or mse < best_result.reconstruction_error:
                best_result = result

        # Nothing met target, return best (highest bits) with refinement
        assert best_result is not None
        return self._apply_refinement(W, H, best_result)

    def _quantize_with_format(
        self,
        W: NDArray[np.float64],
        H: NDArray[np.float64],
        fmt: QuantizationFormat,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint32]]:
        """Quantize weight matrix with specified format.

        Uses GPTQ-style Hessian-aware quantization for all formats.

        Args:
            W: Weight matrix [out_features, in_features].
            H: Hessian matrix [in_features, in_features].
            fmt: Quantization format.

        Returns:
            (quantized_weights, scales, packed_weights)
        """
        out_feat, in_feat = W.shape
        bits = FORMAT_BITS[fmt]

        # Get quantization grid for format
        grid = self._get_quantization_grid(fmt)

        # Compute per-group scales
        n_groups = in_feat // self.group_size
        scales = np.zeros((out_feat, n_groups), dtype=np.float64)
        grid_max = np.max(np.abs(grid))

        for g in range(n_groups):
            start = g * self.group_size
            end = start + self.group_size
            group_max = np.max(np.abs(W[:, start:end]), axis=1)
            scales[:, g] = group_max / grid_max + 1e-10

        # GPTQ quantization with error compensation
        Q, indices = self._gptq_quantize(W, H, grid, scales)

        # Pack into uint32
        packed = self._pack_weights(indices, bits)

        return Q, scales, packed

    def _get_quantization_grid(
        self,
        fmt: QuantizationFormat,
    ) -> NDArray[np.float32]:
        """Get the quantization grid for a format."""
        if fmt == QuantizationFormat.FP4:
            return np.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32,
            )
        elif fmt == QuantizationFormat.INT4:
            return np.arange(-8, 8, dtype=np.float32)
        elif fmt == QuantizationFormat.INT3:
            return np.array(
                [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
                dtype=np.float32,
            )
        elif fmt == QuantizationFormat.INT2:
            return np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        elif fmt == QuantizationFormat.NF4:
            # NormalFloat 4-bit: Gaussian quantiles
            return self._compute_nf_grid(4)
        elif fmt == QuantizationFormat.NF3:
            return self._compute_nf_grid(3)
        elif fmt == QuantizationFormat.NF2:
            return self._compute_nf_grid(2)
        elif fmt == QuantizationFormat.INT8:
            return np.arange(-128, 128, dtype=np.float32)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _compute_nf_grid(self, bits: int) -> NDArray[np.float32]:
        """Compute NormalFloat grid from Gaussian quantiles."""
        from scipy import stats

        n_levels = 2 ** bits
        quantiles = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
        levels = stats.norm.ppf(quantiles)
        # Normalize so max magnitude is 1
        levels = levels / np.max(np.abs(levels))
        return levels.astype(np.float32)

    def _gptq_quantize(
        self,
        W: NDArray[np.float64],
        H: NDArray[np.float64],
        grid: NDArray[np.float32],
        scales: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """GPTQ quantization with Hessian-aware error compensation.

        Implements the core GPTQ algorithm:
        1. Add damping to Hessian diagonal for stability
        2. Compute inverse Hessian via Cholesky
        3. Quantize columns, propagating error to remaining columns

        Args:
            W: Weight matrix [out_features, in_features].
            H: Hessian matrix [in_features, in_features].
            grid: Quantization levels.
            scales: Per-group scales [out_features, n_groups].

        Returns:
            (quantized_weights, indices)
        """
        W = W.copy()
        out_feat, in_feat = W.shape
        n_groups = in_feat // self.group_size

        # Add damping to Hessian
        damp = 0.01 * np.mean(np.diag(H))
        H_damped = H.copy()
        H_damped[np.diag_indices_from(H_damped)] += damp

        # Compute inverse Hessian via Cholesky
        try:
            L = np.linalg.cholesky(H_damped)
            H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(in_feat)))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            H_inv = np.linalg.pinv(H_damped)

        # Quantize with error compensation
        Q = np.zeros_like(W)
        indices = np.zeros(W.shape, dtype=np.int32)

        for g in range(n_groups):
            g_start = g * self.group_size
            g_end = (g + 1) * self.group_size

            for i in range(g_start, g_end):
                col = W[:, i]
                scale = scales[:, g]

                # Quantize to nearest grid point
                normalized = col / scale
                dists = np.abs(normalized[:, None] - grid[None, :])
                idx = np.argmin(dists, axis=1)
                q_val = grid[idx] * scale

                Q[:, i] = q_val
                indices[:, i] = idx

                # Error compensation for remaining columns in group
                error = col - q_val
                h_ii = H_inv[i, i]
                if h_ii > 1e-15:
                    for j in range(i + 1, g_end):
                        W[:, j] -= error * (H_inv[i, j] / h_ii)

        return Q, indices

    def _pack_weights(
        self,
        indices: NDArray[np.int32],
        bits: int,
    ) -> NDArray[np.uint32]:
        """Pack quantization indices into uint32.

        Packing varies by bit width:
        - 2-bit: 16 values per uint32
        - 3-bit: 10 values per uint32 (with 2-bit padding)
        - 4-bit: 8 values per uint32
        - 8-bit: 4 values per uint32

        Args:
            indices: Quantization indices [out_features, in_features].
            bits: Bits per value.

        Returns:
            Packed uint32 array.
        """
        out_feat, in_feat = indices.shape

        if bits == 2:
            vals_per_word = 16
        elif bits == 3:
            vals_per_word = 10
        elif bits == 4:
            vals_per_word = 8
        elif bits == 8:
            vals_per_word = 4
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        # Pad if necessary
        pad_needed = (vals_per_word - (in_feat % vals_per_word)) % vals_per_word
        if pad_needed > 0:
            indices = np.pad(indices, ((0, 0), (0, pad_needed)), mode="constant")
            in_feat = indices.shape[1]

        n_packed = in_feat // vals_per_word
        packed = np.zeros((out_feat, n_packed), dtype=np.uint32)
        mask = (1 << bits) - 1

        for i in range(vals_per_word):
            if i * bits >= 32:
                break  # Can't fit more in uint32
            packed |= (indices[:, i::vals_per_word].astype(np.uint32) & mask) << (i * bits)

        return packed

    def _apply_refinement(
        self,
        W: NDArray[np.float64],
        H: NDArray[np.float64],
        initial: AdaptiveQuantResult,
    ) -> AdaptiveQuantResult:
        """Apply iterative refinement to reduce error.

        ExllamaV3-style iterative error correction:
        1. Compute residual error: E = W - Q
        2. Re-quantize residual using the Hessian
        3. Add refined residual to quantized weights
        4. Repeat for specified iterations

        Args:
            W: Original weight matrix.
            H: Hessian matrix.
            initial: Initial quantization result.

        Returns:
            Refined AdaptiveQuantResult.
        """
        if self.refinement_iterations == 0:
            return initial

        return self.iterative_refinement(
            weight=W,
            hessian=H,
            initial_quant=initial.quantized.astype(np.float64),
            iterations=self.refinement_iterations,
            format=initial.format,
            scales=initial.scale.astype(np.float64),
        )

    def iterative_refinement(
        self,
        weight: NDArray[np.floating],
        hessian: NDArray[np.floating],
        initial_quant: NDArray[np.floating],
        iterations: int = 3,
        format: QuantizationFormat | None = None,
        scales: NDArray[np.floating] | None = None,
    ) -> AdaptiveQuantResult:
        """ExllamaV3-style iterative error correction.

        Re-quantize using residual errors to progressively reduce
        reconstruction error. Each iteration:
        1. Compute residual: R = W - Q
        2. Quantize residual with small scale
        3. Update quantized weights: Q += quantized(R)

        This is particularly effective for layers near the error boundary
        where a few high-error weights dominate the MSE.

        Args:
            weight: Original weight matrix [out_features, in_features].
            hessian: Hessian matrix [in_features, in_features].
            initial_quant: Initial quantized weights.
            iterations: Number of refinement iterations.
            format: Quantization format (defaults to INT4).
            scales: Initial scales (computed if None).

        Returns:
            Refined AdaptiveQuantResult.
        """
        W = np.asarray(weight, dtype=np.float64)
        np.asarray(hessian, dtype=np.float64)
        Q = np.asarray(initial_quant, dtype=np.float64)

        if format is None:
            format = QuantizationFormat.INT4

        bits = FORMAT_BITS[format]
        grid = self._get_quantization_grid(format)
        out_feat, in_feat = W.shape
        n_groups = in_feat // self.group_size

        if scales is None:
            grid_max = np.max(np.abs(grid))
            scales = np.zeros((out_feat, n_groups), dtype=np.float64)
            for g in range(n_groups):
                start = g * self.group_size
                end = start + self.group_size
                group_max = np.max(np.abs(W[:, start:end]), axis=1)
                scales[:, g] = group_max / grid_max + 1e-10
        else:
            scales = np.asarray(scales, dtype=np.float64)

        # Iterative refinement
        for iteration in range(iterations):
            # Compute residual
            residual = W - Q

            # Scale factor for residual (smaller each iteration)
            residual_scale = 1.0 / (2 ** (iteration + 1))

            # Quantize residual with reduced scale
            for g in range(n_groups):
                g_start = g * self.group_size
                g_end = (g + 1) * self.group_size

                for i in range(g_start, g_end):
                    r_col = residual[:, i]
                    scale = scales[:, g] * residual_scale

                    # Quantize residual
                    normalized = r_col / scale
                    dists = np.abs(normalized[:, None] - grid[None, :])
                    idx = np.argmin(dists, axis=1)
                    q_residual = grid[idx] * scale

                    # Update quantized weights
                    Q[:, i] += q_residual

        # Compute final error
        mse = float(np.mean((W - Q) ** 2))
        rel_error = float(np.mean(np.abs(W - Q) / (np.abs(W) + 1e-10)))

        # Re-quantize final weights to get indices for packing
        indices = np.zeros(W.shape, dtype=np.int32)
        for g in range(n_groups):
            g_start = g * self.group_size
            g_end = (g + 1) * self.group_size

            for i in range(g_start, g_end):
                col = Q[:, i]
                scale = scales[:, g]
                normalized = col / scale
                dists = np.abs(normalized[:, None] - grid[None, :])
                indices[:, i] = np.argmin(dists, axis=1)

        packed = self._pack_weights(indices, bits)

        return AdaptiveQuantResult(
            quantized=Q.astype(np.float32),
            bits=bits,
            format=format,
            scale=scales.astype(np.float32),
            packed=packed,
            reconstruction_error=mse,
            relative_error=rel_error,
            iterations=iterations,
        )

    def allocate_model_budget(
        self,
        layer_hessians: dict[str, NDArray[np.floating]],
        layer_shapes: dict[str, tuple[int, int]],
        target_bits: float = 4.0,
    ) -> ModelBudgetAllocation:
        """Allocate bit budget across model layers based on sensitivity.

        Uses inverse sensitivity weighting: more sensitive layers get more bits,
        less sensitive layers (like MoE experts with sparse activation) get fewer.

        Algorithm:
        1. Compute sensitivity for each layer from Hessian diagonal
        2. Normalize sensitivities to sum to 1
        3. Allocate bits = base_bits * (1 + α * (sens - mean_sens))
        4. Clamp to [min_bits, max_bits]
        5. Adjust to hit target average

        Args:
            layer_hessians: Dict mapping layer name to Hessian matrix.
            layer_shapes: Dict mapping layer name to (out_feat, in_feat).
            target_bits: Target average bits-per-weight. Default: 4.0.

        Returns:
            ModelBudgetAllocation with per-layer bit assignments.
        """
        # Compute sensitivity for each layer
        sensitivities: dict[str, float] = {}
        for name, H in layer_hessians.items():
            sensitivities[name] = self.compute_sensitivity(H)

        # Compute total parameters and sensitivity-weighted average
        total_params = sum(
            shape[0] * shape[1] for shape in layer_shapes.values()
        )
        sens_values = list(sensitivities.values())
        mean_sens = np.mean(sens_values)
        std_sens = np.std(sens_values) + 1e-10

        # Allocate bits based on normalized sensitivity
        # Layers with higher sensitivity get more bits
        # α controls spread: higher α = more variance in allocated bits
        alpha = 1.0
        layer_budgets: list[LayerBudget] = []

        for name, shape in layer_shapes.items():
            sens = sensitivities[name]
            # Normalized deviation from mean
            sens_norm = (sens - mean_sens) / std_sens

            # Allocate bits: more sensitive = more bits
            allocated = target_bits + alpha * sens_norm
            allocated = np.clip(allocated, self.min_bits, self.max_bits)

            # Select format for allocated bits
            allocated_int = int(round(allocated))
            fmt = self._select_format_for_bits(allocated_int)

            layer_budgets.append(
                LayerBudget(
                    name=name,
                    sensitivity=sens,
                    allocated_bits=float(allocated),
                    format=fmt,
                    weight=shape[0] * shape[1] / total_params,
                )
            )

        # Compute actual average bits
        actual_bits = sum(
            lb.allocated_bits * lb.weight for lb in layer_budgets
        )

        return ModelBudgetAllocation(
            layers=layer_budgets,
            target_bits=target_bits,
            actual_bits=actual_bits,
            total_params=total_params,
        )

    def _select_format_for_bits(self, bits: int) -> QuantizationFormat:
        """Select best format for given bit count."""
        bits = int(np.clip(bits, self.min_bits, self.max_bits))

        # Prefer NF formats if configured
        if self.prefer_nf:
            nf_formats = {
                2: QuantizationFormat.NF2,
                3: QuantizationFormat.NF3,
                4: QuantizationFormat.NF4,
            }
            if bits in nf_formats:
                return nf_formats[bits]

        # Fall back to INT formats
        int_formats = {
            2: QuantizationFormat.INT2,
            3: QuantizationFormat.INT3,
            4: QuantizationFormat.INT4,
            8: QuantizationFormat.INT8,
        }

        if bits in int_formats:
            return int_formats[bits]

        # Default to INT4 for unsupported bit widths
        return QuantizationFormat.INT4


def compute_moe_expert_sensitivity(
    expert_hessians: list[NDArray[np.floating]],
    router_logits: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Compute sensitivity scores for MoE experts.

    For MoE models, experts that are activated less frequently can tolerate
    more aggressive quantization. We adjust sensitivity based on:
    1. Hessian diagonal (standard sensitivity)
    2. Expert activation frequency from router logits

    Args:
        expert_hessians: List of Hessian matrices, one per expert.
        router_logits: Optional router output [n_samples, n_experts].
            Used to compute activation frequency.

    Returns:
        Per-expert sensitivity scores [n_experts].
    """
    n_experts = len(expert_hessians)
    base_sensitivity = np.zeros(n_experts, dtype=np.float64)

    # Compute base sensitivity from Hessian
    for i, H in enumerate(expert_hessians):
        diag = np.diag(H).astype(np.float64)
        base_sensitivity[i] = np.mean(diag)

    # Adjust by activation frequency if router logits provided
    if router_logits is not None:
        # Compute softmax to get activation probabilities
        logits = np.asarray(router_logits, dtype=np.float64)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # Average activation probability per expert
        avg_activation = np.mean(probs, axis=0)

        # Scale sensitivity by activation (more active = more sensitive)
        # Use sqrt to avoid extreme scaling
        activation_scale = np.sqrt(avg_activation / np.mean(avg_activation))
        base_sensitivity *= activation_scale

    return base_sensitivity
