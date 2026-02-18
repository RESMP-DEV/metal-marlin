"""Per-layer FLOPs calculator for Metal Marlin layers.

Provides LayerFLOPsCalculator for computing FLOPs on actual PyTorch modules
and specialized functions for estimating FLOPs of quantized operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metal_marlin.utils.profile_ops import (
    LayerFLOPsCounter,
    calculate_attention_flops,
    calculate_ffn_flops,
    calculate_layernorm_flops,
    calculate_matmul_flops,
)


@dataclass
class LayerProfileConfig:
    """Configuration for layer FLOPs profiling.

    Attributes:
        batch_size: Default batch size for profiling.
        seq_len: Default sequence length for profiling.
        quantized: Whether to account for quantization overhead.
        group_size: Quantization group size (for quantized layers).
        causal_attention: Use causal masking for attention.
        gated_ffn: Use gated FFN (SwiGLU) for FFN layers.
    """

    batch_size: int = 1
    seq_len: int = 2048
    quantized: bool = True
    group_size: int = 128
    causal_attention: bool = True
    gated_ffn: bool = True


@dataclass
class ModuleFLOPsResult:
    """Result of FLOPs calculation for a module.

    Attributes:
        name: Module name/path.
        module_type: Type of module (e.g., "MarlinLinear", "attention").
        flops: Total FLOPs.
        params: Number of parameters.
        input_shape: Input tensor shape.
        output_shape: Output tensor shape.
        metadata: Additional profiling metadata.
    """

    name: str
    module_type: str
    flops: int
    params: int = 0
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tflops(self) -> float:
        """FLOPs in trillions."""
        return self.flops / 1e12

    @property
    def gflops(self) -> float:
        """FLOPs in billions."""
        return self.flops / 1e9


def estimate_marlin_linear_flops(
    in_features: int,
    out_features: int,
    batch_size: int = 1,
    seq_len: int = 1,
    quantized: bool = True,
    group_size: int = 128,
    include_bias: bool = True,
) -> int:
    """Estimate FLOPs for a MarlinLinear layer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        batch_size: Batch size.
        seq_len: Sequence length (tokens per batch).
        quantized: Account for FP4 dequantization overhead.
        group_size: Quantization group size.
        include_bias: Include bias addition FLOPs.

    Returns:
        Total FLOPs for the linear operation.

    Example:
        >>> flops = estimate_marlin_linear_flops(
        ...     in_features=4096,
        ...     out_features=11008,
        ...     batch_size=8,
        ...     seq_len=2048,
        ...     quantized=True
        ... )
        >>> print(f"{flops / 1e12:.2f} TFLOPs")
    """
    tokens = batch_size * seq_len

    # Main GEMM: (tokens, in_features) @ (in_features, out_features)
    gemm_flops = calculate_matmul_flops(tokens, out_features, in_features, quantized=quantized)

    # Bias addition: 1 op per output element
    bias_flops = tokens * out_features if include_bias else 0

    # Dequantization overhead for FP4
    # For each output element, we need to dequantize from FP4
    # This involves: unpack nibble, lookup in E2M1 table, apply scale
    dequant_flops = 0
    if quantized:
        # ~4 ops per weight element for FP4 dequantization
        n_groups = (in_features + group_size - 1) // group_size
        dequant_flops = tokens * out_features * n_groups * 4

    return gemm_flops + bias_flops + dequant_flops


def calculate_layer_flops(
    layer_type: str,
    **kwargs: Any,
) -> int:
    """Calculate FLOPs for a layer by type.

    Args:
        layer_type: Type of layer ("linear", "attention", "ffn", "layernorm").
        **kwargs: Layer-specific arguments.

    Returns:
        Total FLOPs for the layer.

    Raises:
        ValueError: If layer_type is not recognized.

    Example:
        >>> # Linear layer
        >>> flops = calculate_layer_flops(
        ...     "linear",
        ...     in_features=4096,
        ...     out_features=11008,
        ...     batch_size=8,
        ...     seq_len=2048
        ... )

        >>> # Attention layer
        >>> flops = calculate_layer_flops(
        ...     "attention",
        ...     batch=8,
        ...     seq_len=2048,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     causal=True
        ... )

        >>> # FFN layer
        >>> flops = calculate_layer_flops(
        ...     "ffn",
        ...     batch=8,
        ...     seq_len=2048,
        ...     hidden_dim=4096,
        ...     ffn_dim=11008,
        ...     gated=True
        ... )
    """
    layer_type = layer_type.lower()

    if layer_type == "linear":
        return estimate_marlin_linear_flops(
            in_features=kwargs["in_features"],
            out_features=kwargs["out_features"],
            batch_size=kwargs.get("batch_size", 1),
            seq_len=kwargs.get("seq_len", 1),
            quantized=kwargs.get("quantized", True),
            group_size=kwargs.get("group_size", 128),
            include_bias=kwargs.get("include_bias", True),
        )

    elif layer_type == "attention":
        return calculate_attention_flops(
            batch=kwargs["batch"],
            seq_len=kwargs["seq_len"],
            num_heads=kwargs["num_heads"],
            head_dim=kwargs["head_dim"],
            causal=kwargs.get("causal", True),
        )

    elif layer_type == "ffn":
        return calculate_ffn_flops(
            batch=kwargs["batch"],
            seq_len=kwargs["seq_len"],
            hidden_dim=kwargs["hidden_dim"],
            ffn_dim=kwargs["ffn_dim"],
            gated=kwargs.get("gated", True),
            quantized=kwargs.get("quantized", True),
        )

    elif layer_type in ("layernorm", "rmsnorm", "norm"):
        return calculate_layernorm_flops(
            batch=kwargs.get("batch", 1),
            seq_len=kwargs.get("seq_len", 1),
            hidden_dim=kwargs.get("hidden_dim", kwargs.get("in_features", 1)),
        )

    else:
        raise ValueError(f"Unknown layer_type: {layer_type!r}")


class LayerFLOPsCalculator:
    """Calculate FLOPs for PyTorch modules with Metal Marlin support.

    This calculator can analyze nn.Module instances and compute their
    theoretical FLOPs, with special handling for MarlinLinear and other
    quantized layers.

    Example:
        >>> calculator = LayerFLOPsCalculator(batch_size=8, seq_len=2048)
        >>> for name, module in model.named_modules():
        ...     calculator.add_module(name, module)
        >>> calculator.print_summary()
        >>> total = calculator.total_flops
    """

    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 2048,
        quantized: bool = True,
        group_size: int = 128,
    ):
        """Initialize the calculator.

        Args:
            batch_size: Default batch size for calculations.
            seq_len: Default sequence length.
            quantized: Account for quantization overhead.
            group_size: Quantization group size.
        """
        self.config = LayerProfileConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            quantized=quantized,
            group_size=group_size,
        )
        self._results: list[ModuleFLOPsResult] = []
        self._counter = LayerFLOPsCounter()

    def add_module(
        self,
        name: str,
        module: Any,
        input_shape: tuple[int, ...] | None = None,
    ) -> ModuleFLOPsResult | None:
        """Add a PyTorch module to the calculator.

        Args:
            name: Module name/path.
            module: The PyTorch module to analyze.
            input_shape: Optional input shape override.

        Returns:
            ModuleFLOPsResult if the module was recognized, None otherwise.
        """
        try:
            result = self._analyze_module(name, module, input_shape)
            if result:
                self._results.append(result)
            return result
        except Exception:
            # Skip modules that can't be analyzed
            return None

    def _analyze_module(
        self,
        name: str,
        module: Any,
        input_shape: tuple[int, ...] | None = None,
    ) -> ModuleFLOPsResult | None:
        """Analyze a single module and compute its FLOPs."""
        module_type = type(module).__name__

        # Handle MarlinLinear
        if module_type in ("MarlinLinear", "MMFP4Linear"):
            return self._analyze_marlin_linear(name, module)

        # Handle standard Linear
        if module_type == "Linear":
            return self._analyze_linear(name, module)

        # Skip other modules for now
        return None

    def _analyze_marlin_linear(
        self,
        name: str,
        module: Any,
    ) -> ModuleFLOPsResult:
        """Analyze a MarlinLinear module."""
        in_features = getattr(module, "in_features", 0)
        out_features = getattr(module, "out_features", 0)
        group_size = getattr(module, "group_size", self.config.group_size)
        has_bias = getattr(module, "bias", None) is not None

        flops = estimate_marlin_linear_flops(
            in_features=in_features,
            out_features=out_features,
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
            quantized=self.config.quantized,
            group_size=group_size,
            include_bias=has_bias,
        )

        # Count parameters
        params = 0
        if hasattr(module, "weight_packed"):
            params += module.weight_packed.numel() * 4 / 8  # 4 bits per element
        if hasattr(module, "scales"):
            params += module.scales.numel() * 2  # 16 bits per scale
        if has_bias and hasattr(module, "bias"):
            params += module.bias.numel() * 2  # 16 bits per bias

        return ModuleFLOPsResult(
            name=name,
            module_type=type(module).__name__,
            flops=flops,
            params=int(params),
            input_shape=(self.config.batch_size, self.config.seq_len, in_features),
            output_shape=(self.config.batch_size, self.config.seq_len, out_features),
            metadata={
                "in_features": in_features,
                "out_features": out_features,
                "group_size": group_size,
                "quantized": True,
            },
        )

    def _analyze_linear(
        self,
        name: str,
        module: Any,
    ) -> ModuleFLOPsResult:
        """Analyze a standard Linear module."""
        in_features = getattr(module, "in_features", 0)
        out_features = getattr(module, "out_features", 0)
        has_bias = getattr(module, "bias", None) is not None

        flops = estimate_marlin_linear_flops(
            in_features=in_features,
            out_features=out_features,
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
            quantized=False,
            include_bias=has_bias,
        )

        params = 0
        if hasattr(module, "weight"):
            params += module.weight.numel() * 2  # Assume FP16
        if has_bias and hasattr(module, "bias"):
            params += module.bias.numel() * 2

        return ModuleFLOPsResult(
            name=name,
            module_type="Linear",
            flops=flops,
            params=int(params),
            input_shape=(self.config.batch_size, self.config.seq_len, in_features),
            output_shape=(self.config.batch_size, self.config.seq_len, out_features),
            metadata={
                "in_features": in_features,
                "out_features": out_features,
                "quantized": False,
            },
        )

    @property
    def total_flops(self) -> int:
        """Total FLOPs across all analyzed modules."""
        return sum(r.flops for r in self._results)

    @property
    def total_params(self) -> int:
        """Total parameters across all analyzed modules."""
        return sum(r.params for r in self._results)

    @property
    def total_tflops(self) -> float:
        """Total FLOPs in trillions."""
        return self.total_flops / 1e12

    def get_results(self) -> list[ModuleFLOPsResult]:
        """Get all profiling results."""
        return list(self._results)

    def get_layer(self, name: str) -> ModuleFLOPsResult | None:
        """Get result for a specific layer by name."""
        for r in self._results:
            if r.name == name:
                return r
        return None

    def print_summary(self, top_n: int = 20) -> None:
        """Print a formatted summary of FLOPs by layer.

        Args:
            top_n: Number of top layers to display.
        """
        if not self._results:
            print("No layers profiled")
            return

        sorted_results = sorted(self._results, key=lambda x: x.flops, reverse=True)

        print(f"{'Layer':<50} {'Type':<15} {'TFLOPs':>10} {'Params(M)':>10} {'% Total':>8}")
        print("-" * 95)

        total = self.total_flops
        for r in sorted_results[:top_n]:
            pct = (r.flops / total) * 100 if total > 0 else 0
            params_m = r.params / 1e6
            print(f"{r.name:<50} {r.module_type:<15} {r.tflops:>10.3f} {params_m:>10.1f} {pct:>7.1f}%")

        print("-" * 95)
        print(f"{'TOTAL':<50} {'':<15} {self.total_tflops:>10.3f} {self.total_params/1e6:>10.1f} {'100.0':>7}%")

        if len(sorted_results) > top_n:
            print(f"\n(Showing top {top_n} of {len(sorted_results)} layers)")

    def clear(self) -> None:
        """Clear all profiling results."""
        self._results.clear()
        self._counter.clear()


def profile_model_layers(
    model: Any,
    batch_size: int = 1,
    seq_len: int = 2048,
    quantized: bool = True,
) -> LayerFLOPsCalculator:
    """Profile all layers in a model.

    Convenience function to profile an entire model.

    Args:
        model: PyTorch model to profile.
        batch_size: Batch size for calculations.
        seq_len: Sequence length.
        quantized: Account for quantization overhead.

    Returns:
        LayerFLOPsCalculator with profiling results.

    Example:
        >>> calculator = profile_model_layers(model, batch_size=8, seq_len=2048)
        >>> calculator.print_summary()
        >>> print(f"Total: {calculator.total_tflops:.2f} TFLOPs")
    """
    calculator = LayerFLOPsCalculator(
        batch_size=batch_size,
        seq_len=seq_len,
        quantized=quantized,
    )

    for name, module in model.named_modules():
        # Skip the root module
        if name == "":
            continue
        calculator.add_module(name, module)

    return calculator
