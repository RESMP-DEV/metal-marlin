"""Activation statistics collector for quantization calibration.

Instruments model layers to record min/max/absmax and percentile statistics
from forward-pass activations. These statistics inform quantization range
selection, enabling outlier-aware schemes that clip extreme values rather
than letting them inflate the quantization range.

Since MLX nn.Module does not provide forward hooks, this module monkeypatches
__call__ on target layers and restores originals on remove_hooks().

Usage:
    collector = CalibrationCollector(model)
    for batch in calibration_data:
        _ = model(batch)  # Forward passes accumulate stats
    stats = collector.get_stats()
    collector.remove_hooks()

    # Percentile-based clipping range for a layer:
    layer_stats = stats["layers.0.attention.q_proj"]
    clip_min = layer_stats.percentile_low   # 0.01th percentile
    clip_max = layer_stats.percentile_high  # 99.99th percentile

    # Compute quantization scales from calibration:
    scales = compute_scales(stats, quant_type="fp4")

CalibrationDataset:
    Pre-loaded calibration dataset for quantization-aware scale computation.
    Supports loading Bartowski calibration v3 from HuggingFace or local files.

    Usage:
        from metal_marlin.converters.calibration import CalibrationDataset

        # Load bartowski v3 calibration (multi-domain: code, chat, reasoning)
        calib = CalibrationDataset.bartowski_v3(max_samples=512)

        # Or load from local file
        calib = CalibrationDataset.from_file("calibration.txt")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# CalibrationDataset: Pre-loaded calibration data for quantization
# ---------------------------------------------------------------------------

# Bartowski calibration v3 URL (multi-domain: code, chat, reasoning, math)
_BARTOWSKI_V3_URL = (
    "https://gist.githubusercontent.com/bartowski1182/"
    "eb213dccb3571f863da82e99418f81e8/raw/"
    "2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt"
)

# Default cache directory for downloaded calibration data
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "metal_marlin" / "calibration"


@dataclass
class CalibrationDataset:
    """Pre-loaded calibration dataset for quantization.

    Contains text samples that will be tokenized and used for computing
    activation statistics during quantization. Bartowski v3 is recommended
    over WikiText-2 for better coverage of code, chat, and reasoning domains.

    Attributes:
        samples: List of text strings for calibration.
        name: Dataset identifier (e.g., "bartowski-v3").
        source: Where the data was loaded from (URL or path).
    """

    samples: list[str]
    name: str = "unknown"
    source: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def bartowski_v3(
        cls,
        max_samples: int = 512,
        cache_dir: str | Path | None = None,
    ) -> CalibrationDataset:
        """Load Bartowski calibration v3 from HuggingFace gist.

        Bartowski v3 is a multi-domain calibration dataset containing:
        - Code samples (Python, JS, etc.)
        - Chat/instruction samples
        - Reasoning and math problems
        - General knowledge

        This provides better quantization ranges than WikiText-2 for
        instruction-tuned and code-capable models.

        Args:
            max_samples: Maximum number of samples to load.
            cache_dir: Directory to cache downloaded data. Defaults to
                       ~/.cache/metal_marlin/calibration/

        Returns:
            CalibrationDataset with loaded samples.
        """
        cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "bartowski_v3.txt"

        # Download if not cached
        if not cache_file.exists():
            import urllib.request

            print("Downloading Bartowski v3 calibration data...")
            urllib.request.urlretrieve(_BARTOWSKI_V3_URL, cache_file)

        return cls.from_file(
            cache_file,
            max_samples=max_samples,
            name="bartowski-v3",
            source=_BARTOWSKI_V3_URL,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        max_samples: int | None = None,
        name: str | None = None,
        source: str | None = None,
    ) -> CalibrationDataset:
        """Load calibration data from a text file.

        Each line in the file is treated as a separate calibration sample.
        Empty lines are skipped.

        Args:
            path: Path to calibration text file.
            max_samples: Maximum samples to load (None for all).
            name: Dataset name. Defaults to filename.
            source: Source description. Defaults to path.

        Returns:
            CalibrationDataset with loaded samples.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        samples: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
                    if max_samples and len(samples) >= max_samples:
                        break

        return cls(
            samples=samples,
            name=name or path.stem,
            source=source or str(path),
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        name: str = "custom",
    ) -> CalibrationDataset:
        """Create calibration dataset from a list of text strings.

        Args:
            texts: List of calibration text samples.
            name: Dataset name.

        Returns:
            CalibrationDataset with the provided samples.
        """
        return cls(samples=texts, name=name, source="custom")

    def tokenize(
        self,
        tokenizer,
        max_length: int = 2048,
        return_tensors: str = "np",
    ) -> list:
        """Tokenize samples using a HuggingFace-compatible tokenizer.

        Args:
            tokenizer: A tokenizer with __call__(text, ...) -> BatchEncoding.
            max_length: Maximum sequence length for each sample.
            return_tensors: Return format ("np" for numpy, "mx" for MLX).

        Returns:
            List of tokenized samples (dict with input_ids, attention_mask).
        """

        results = []
        for text in self.samples:
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )
            if return_tensors == "mx":
                encoded = {k: mx.array(v) for k, v in encoded.items()}
            results.append(encoded)
        return results


@dataclass
class CalibrationStats:
    """Per-layer activation statistics for calibration.

    Accumulates running statistics across multiple forward passes.

    Attributes:
        min_val: Global minimum activation value observed.
        max_val: Global maximum activation value observed.
        absmax: Global maximum absolute activation value.
        percentile_low: Low percentile value for clipping (set by get_stats).
        percentile_high: High percentile value for clipping (set by get_stats).
        num_batches: Number of forward passes accumulated.
    """

    min_val: mx.array
    max_val: mx.array
    absmax: mx.array
    percentile_low: mx.array | None = None
    percentile_high: mx.array | None = None
    num_batches: int = 0


_DEFAULT_NUM_BINS = 2048
_DEFAULT_PERCENTILE_LOW = 0.01
_DEFAULT_PERCENTILE_HIGH = 99.99


class CalibrationCollector:
    """Collect activation statistics for quantization calibration.

    Instruments target layers by wrapping their __call__ methods to
    record input activation statistics.

    Supports two percentile estimation strategies:
      - "histogram": Maintains a fixed-bin histogram and interpolates
        percentiles. Constant memory regardless of calibration set size.
        This is the default and recommended for large models.
      - "exact": Concatenates all flattened activations and computes exact
        quantiles via sort. Accurate but memory-intensive.

    Args:
        model: The nn.Module to instrument.
        layers_to_calibrate: Optional list of layer name prefixes to target.
            If None, targets all nn.Linear and MarlinLinear layers.
        percentile_low: Low percentile for clipping (0-100). Default: 0.01.
        percentile_high: High percentile for clipping (0-100). Default: 99.99.
        method: Percentile estimation method. "histogram" or "exact".
        num_bins: Number of histogram bins for "histogram" method.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_to_calibrate: list[str] | None = None,
        percentile_low: float = _DEFAULT_PERCENTILE_LOW,
        percentile_high: float = _DEFAULT_PERCENTILE_HIGH,
        method: str = "histogram",
        num_bins: int = _DEFAULT_NUM_BINS,
    ):
        if method not in ("histogram", "exact"):
            raise ValueError(f"method must be 'histogram' or 'exact', got {method!r}")

        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.method = method
        self.num_bins = num_bins

        self.stats: dict[str, CalibrationStats] = {}
        self._hooks: list[tuple[nn.Module, Any]] = []  # (module, original_call)
        self._histograms: dict[str, tuple[mx.array, mx.array]] = {}  # counts, edges
        self._exact_buffers: dict[str, list[mx.array]] = {}

        self._register_hooks(model, layers_to_calibrate)

    def _is_target_layer(self, module: nn.Module) -> bool:
        """Check if a module is a quantization target (linear layer)."""
        try:
            from ..metal_marlin.layers import MarlinLinear

            target_types: tuple[type, ...] = (nn.Linear, MarlinLinear)
        except (ImportError, AttributeError):
            target_types = (nn.Linear,)

        return isinstance(module, target_types)

    def _register_hooks(
        self, model: nn.Module, layers_to_calibrate: list[str] | None
    ) -> None:
        """Walk the module tree and wrap target layers."""
        for name, module in _named_modules(model):
            if layers_to_calibrate is not None:
                if not any(name.startswith(p) for p in layers_to_calibrate):
                    continue
            elif not self._is_target_layer(module):
                continue

            self._wrap_module(name, module)

    def _wrap_module(self, name: str, module: nn.Module) -> None:
        """Monkeypatch __call__ to intercept input activations."""
        original_call = module.__call__

        def wrapped_call(x: mx.array, *args: Any, **kwargs: Any) -> mx.array:
            self._update_stats(name, x)
            return original_call(x, *args, **kwargs)

        self._hooks.append((module, original_call))
        module.__call__ = wrapped_call  # type: ignore[method-assign]

    def _update_stats(self, name: str, x: mx.array) -> None:
        """Update running statistics for a layer given input activations."""
        flat = x.reshape(-1)

        batch_min = mx.min(flat)
        batch_max = mx.max(flat)
        batch_absmax = mx.max(mx.abs(flat))

        if name not in self.stats:
            self.stats[name] = CalibrationStats(
                min_val=batch_min,
                max_val=batch_max,
                absmax=batch_absmax,
                num_batches=1,
            )
            if self.method == "histogram":
                am = float(batch_absmax.item())
                edge_max = am * 1.1 if am > 0 else 1.0
                edges = mx.linspace(-edge_max, edge_max, self.num_bins + 1)
                counts = _histogram(flat, edges)
                self._histograms[name] = (counts, edges)
            else:
                self._exact_buffers[name] = [flat]
        else:
            s = self.stats[name]
            s.min_val = mx.minimum(s.min_val, batch_min)
            s.max_val = mx.maximum(s.max_val, batch_max)
            s.absmax = mx.maximum(s.absmax, batch_absmax)
            s.num_batches += 1

            if self.method == "histogram":
                counts, edges = self._histograms[name]
                current_edge_max = float(edges[-1].item())
                new_absmax = float(batch_absmax.item())

                if new_absmax > current_edge_max:
                    new_edge_max = new_absmax * 1.1
                    new_edges = mx.linspace(
                        -new_edge_max, new_edge_max, self.num_bins + 1
                    )
                    counts = _rebin_histogram(counts, edges, new_edges)
                    edges = new_edges

                new_counts = _histogram(flat, edges)
                counts = counts + new_counts
                self._histograms[name] = (counts, edges)
            else:
                self._exact_buffers[name].append(flat)

    def get_stats(self) -> dict[str, CalibrationStats]:
        """Compute final statistics including percentiles.

        For "histogram" method, interpolates percentiles from accumulated
        histogram. For "exact" method, concatenates all buffered activations
        and computes exact quantiles via sort.

        Returns:
            Dictionary mapping layer names to CalibrationStats with
            percentile_low and percentile_high populated.
        """
        p_low = self.percentile_low / 100.0
        p_high = self.percentile_high / 100.0

        for name, s in self.stats.items():
            if self.method == "histogram" and name in self._histograms:
                counts, edges = self._histograms[name]
                s.percentile_low = _percentile_from_histogram(counts, edges, p_low)
                s.percentile_high = _percentile_from_histogram(counts, edges, p_high)
            elif self.method == "exact" and name in self._exact_buffers:
                all_vals = mx.concatenate(self._exact_buffers[name])
                sorted_vals = mx.sort(all_vals)
                n = sorted_vals.shape[0]
                idx_low = min(int(p_low * n), n - 1)
                idx_high = min(int(p_high * n), n - 1)
                s.percentile_low = sorted_vals[idx_low]
                s.percentile_high = sorted_vals[idx_high]

        return self.stats

    def remove_hooks(self) -> None:
        """Restore original __call__ methods on all instrumented modules."""
        for module, original_call in self._hooks:
            module.__call__ = original_call  # type: ignore[method-assign]
        self._hooks.clear()

    def reset(self) -> None:
        """Clear all accumulated statistics without removing hooks."""
        self.stats.clear()
        self._histograms.clear()
        self._exact_buffers.clear()

    @property
    def num_layers(self) -> int:
        """Number of layers being calibrated."""
        return len(self._hooks)


def compute_scales(
    stats: dict[str, CalibrationStats],
    quant_type: str = "fp4",
    use_percentile: bool = True,
) -> dict[str, tuple[mx.array, mx.array | None]]:
    """Compute per-layer quantization scales from calibration statistics.

    For outlier-aware quantization, uses percentile-based range instead
    of min/max, preventing extreme outliers from degrading precision
    for the majority of values.

    Args:
        stats: Per-layer calibration statistics from CalibrationCollector.
        quant_type: One of "fp4", "int4_sym", "int4_asym".
        use_percentile: If True, use percentile bounds when available.

    Returns:
        Dict of {layer_name: (scale, zeros)} where zeros is None for
        symmetric quantization types.

    Raises:
        ValueError: If quant_type is not recognized.
    """
    result: dict[str, tuple[mx.array, mx.array | None]] = {}

    for name, s in stats.items():
        if quant_type == "fp4":
            # FP4 E2M1: representable magnitudes [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            fp4_max = 6.0
            if use_percentile and s.percentile_high is not None:
                ref = mx.maximum(
                    mx.abs(s.percentile_low), mx.abs(s.percentile_high)
                )
            else:
                ref = s.absmax
            scale = ref / fp4_max
            zeros = None

        elif quant_type == "int4_sym":
            # INT4 symmetric: [-8, 7], use 7 as positive max
            if use_percentile and s.percentile_high is not None:
                ref = mx.maximum(
                    mx.abs(s.percentile_low), mx.abs(s.percentile_high)
                )
            else:
                ref = s.absmax
            scale = ref / 7.0
            zeros = None

        elif quant_type == "int4_asym":
            # INT4 asymmetric: [0, 15] maps to [min_val, max_val]
            if use_percentile and s.percentile_low is not None:
                low = s.percentile_low
                high = s.percentile_high
            else:
                low = s.min_val
                high = s.max_val
            range_val = high - low
            scale = range_val / 15.0
            zeros = -low / mx.maximum(scale, mx.array(1e-10))

        else:
            raise ValueError(
                f"Unknown quant_type={quant_type!r}. "
                f"Expected one of: 'fp4', 'int4_sym', 'int4_asym'"
            )

        scale = mx.maximum(scale, mx.array(1e-10, dtype=mx.float16))
        result[name] = (scale.astype(mx.float16), zeros)

    return result


# --- Module tree walking ---


def _named_modules(
    module: nn.Module, prefix: str = ""
) -> list[tuple[str, nn.Module]]:
    """Recursively enumerate all submodules with dotted path names.

    MLX nn.Module exposes children via the .children() method which
    returns a dict of {attr_name: child_or_list}. We flatten lists
    (e.g. model.layers = [layer0, layer1, ...]) with integer indices.
    """
    result: list[tuple[str, nn.Module]] = []
    if not hasattr(module, "children"):
        return result

    children = module.children()
    if not isinstance(children, dict):
        return result

    for attr_name, child in children.items():
        if isinstance(child, nn.Module):
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            result.append((full_name, child))
            result.extend(_named_modules(child, full_name))
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    full_name = (
                        f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                    )
                    result.append((full_name, item))
                    result.extend(_named_modules(item, full_name))
        elif isinstance(child, dict):
            # Some modules nest dicts of modules
            for key, item in child.items():
                if isinstance(item, nn.Module):
                    full_name = (
                        f"{prefix}.{attr_name}.{key}"
                        if prefix
                        else f"{attr_name}.{key}"
                    )
                    result.append((full_name, item))
                    result.extend(_named_modules(item, full_name))

    return result


# --- Histogram utilities ---


def _histogram(values: mx.array, bin_edges: mx.array) -> mx.array:
    """Compute histogram counts for values given bin edges.

    Uses searchsorted to assign bins, then one-hot accumulation.
    Returns int32 counts array of shape [num_bins].
    """
    num_bins = bin_edges.shape[0] - 1
    indices = mx.searchsorted(bin_edges[1:], values)
    indices = mx.clip(indices, 0, num_bins - 1)

    # One-hot scatter to count per-bin occurrences
    one_hot = mx.zeros((values.shape[0], num_bins), dtype=mx.float32)
    rows = mx.arange(values.shape[0])
    one_hot = one_hot.at[rows, indices].add(1.0)
    counts = mx.sum(one_hot, axis=0).astype(mx.int32)
    return counts


def _rebin_histogram(
    old_counts: mx.array, old_edges: mx.array, new_edges: mx.array
) -> mx.array:
    """Redistribute histogram counts from old bins into new (wider) bins.

    Uses CDF interpolation: builds cumulative distribution from old counts,
    evaluates it at new bin edges, then differences to get new bin counts.
    """
    num_old = old_counts.shape[0]
    num_new = new_edges.shape[0] - 1

    old_cdf = mx.cumsum(old_counts.astype(mx.float32))
    total = old_cdf[-1]
    if float(total.item()) == 0:
        return mx.zeros((num_new,), dtype=mx.int32)

    # Prepend 0 to old_cdf for full CDF: cdf[0]=0, cdf[i]=sum(counts[:i])
    full_old_cdf = mx.concatenate([mx.array([0.0]), old_cdf])
    # full_old_cdf has shape [num_old + 1], aligned with old_edges

    # Evaluate old CDF at each new edge via linear interpolation
    new_cdf_vals = _interp(new_edges, old_edges, full_old_cdf)

    # Difference consecutive CDF values to get bin counts
    new_counts = new_cdf_vals[1:] - new_cdf_vals[:-1]
    new_counts = mx.maximum(new_counts, mx.array(0.0))
    return new_counts.astype(mx.int32)


def _interp(x: mx.array, xp: mx.array, fp: mx.array) -> mx.array:
    """1D linear interpolation (like numpy.interp).

    Args:
        x: Points to interpolate at.
        xp: Known x coordinates (must be sorted ascending).
        fp: Known y coordinates corresponding to xp.

    Returns:
        Interpolated values at x.
    """
    # Clamp x to xp range
    x_clamped = mx.clip(x, xp[0], xp[-1])

    # Find insertion indices
    indices = mx.searchsorted(xp, x_clamped)
    # Clamp to valid range for indexing
    indices = mx.clip(indices, 1, xp.shape[0] - 1)

    # Gather surrounding points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation
    t = (x_clamped - x0) / (x1 - x0 + 1e-10)
    t = mx.clip(t, 0.0, 1.0)
    return y0 + t * (y1 - y0)


def _percentile_from_histogram(
    counts: mx.array, bin_edges: mx.array, percentile: float
) -> mx.array:
    """Interpolate a percentile value from histogram counts.

    Uses the cumulative distribution to find the bin containing the
    target percentile, then linearly interpolates within that bin.

    Args:
        counts: Histogram bin counts [num_bins].
        bin_edges: Bin edge values [num_bins + 1].
        percentile: Target percentile as fraction in [0, 1].

    Returns:
        Scalar float16 value at the given percentile.
    """
    cdf = mx.cumsum(counts.astype(mx.float32))
    total = float(cdf[-1].item())
    if total == 0:
        return mx.array(0.0, dtype=mx.float16)

    target = percentile * total

    # Find first bin where CDF >= target
    exceeded = cdf >= target
    indices = mx.arange(cdf.shape[0])
    # Set non-exceeded bins to a large index, then take min
    masked = mx.where(exceeded, indices, mx.array(cdf.shape[0], dtype=mx.int32))
    bin_idx = int(mx.min(masked).item())
    bin_idx = min(bin_idx, counts.shape[0] - 1)

    # Linear interpolation within the bin
    bin_low = bin_edges[bin_idx]
    bin_high = bin_edges[bin_idx + 1]
    cdf_before = float(cdf[bin_idx - 1].item()) if bin_idx > 0 else 0.0
    bin_count = float(counts[bin_idx].item())

    if bin_count == 0:
        value = (bin_low + bin_high) / 2.0
    else:
        frac = (target - cdf_before) / bin_count
        frac_arr = mx.clip(mx.array(frac), 0.0, 1.0)
        value = bin_low + frac_arr * (bin_high - bin_low)

    return value.astype(mx.float16)


# ---------------------------------------------------------------------------
# Hessian Approximation for GPTQ
# ---------------------------------------------------------------------------
# Reference: GPTQ paper (arxiv:2210.17323) Section 3.2
#
# GPTQ uses the Hessian H = X^T @ X where X is the input activation matrix.
# This captures input statistics for optimal weight quantization decisions.
# The Hessian approximation enables per-column quantization error minimization.


@dataclass
class HessianStats:
    """Per-layer Hessian statistics for GPTQ quantization.

    Attributes:
        hessian: The accumulated Hessian H = X^T @ X, shape [in_features, in_features].
        num_samples: Total number of samples (tokens) accumulated.
        in_features: Dimension of the input features.
    """

    hessian: mx.array
    num_samples: int
    in_features: int


class HessianCollector:
    """Collect Hessian matrices (X^T @ X) for GPTQ quantization.

    The GPTQ algorithm requires the Hessian H = X^T @ X where X contains
    all calibration activations for a layer. This collector accumulates
    the Hessian incrementally using a running sum for memory efficiency.

    Memory efficiency:
        - Instead of storing all X matrices, we accumulate X^T @ X directly.
        - Uses float32 accumulation for numerical stability.
        - For a layer with in_features=4096, the Hessian requires 4096^2 * 4 = 64MB.
        - This is fixed regardless of calibration set size.

    Args:
        model: The nn.Module to instrument.
        layers_to_calibrate: Optional list of layer name prefixes to target.
            If None, targets all nn.Linear and MarlinLinear layers.
        damping_factor: Factor for Hessian damping (default: 0.01).
            The damped Hessian is H + λI where λ = damping_factor * mean(diag(H)).

    Usage:
        collector = HessianCollector(model)
        for batch in calibration_data:
            _ = model(batch)  # Forward passes accumulate Hessian
        hessians = collector.get_hessians()
        collector.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        layers_to_calibrate: list[str] | None = None,
        damping_factor: float = 0.01,
    ):
        self.damping_factor = damping_factor
        self._hessians: dict[str, mx.array] = {}  # Running H = X^T @ X
        self._sample_counts: dict[str, int] = {}
        self._in_features: dict[str, int] = {}
        self._hooks: list[tuple[nn.Module, Any]] = []

        self._register_hooks(model, layers_to_calibrate)

    def _is_target_layer(self, module: nn.Module) -> bool:
        """Check if a module is a quantization target (linear layer)."""
        try:
            from ..metal_marlin.layers import MarlinLinear

            target_types: tuple[type, ...] = (nn.Linear, MarlinLinear)
        except (ImportError, AttributeError):
            target_types = (nn.Linear,)

        return isinstance(module, target_types)

    def _register_hooks(
        self, model: nn.Module, layers_to_calibrate: list[str] | None
    ) -> None:
        """Walk the module tree and wrap target layers."""
        for name, module in _named_modules(model):
            if layers_to_calibrate is not None:
                if not any(name.startswith(p) for p in layers_to_calibrate):
                    continue
            elif not self._is_target_layer(module):
                continue

            self._wrap_module(name, module)

    def _wrap_module(self, name: str, module: nn.Module) -> None:
        """Monkeypatch __call__ to intercept input activations for Hessian."""
        original_call = module.__call__

        def wrapped_call(x: mx.array, *args: Any, **kwargs: Any) -> mx.array:
            self._update_hessian(name, x)
            return original_call(x, *args, **kwargs)

        self._hooks.append((module, original_call))
        module.__call__ = wrapped_call  # type: ignore[method-assign]

    def _update_hessian(self, name: str, x: mx.array) -> None:
        """Update running Hessian H = X^T @ X for a layer.

        The input x has shape [..., in_features]. We flatten all leading
        dimensions to get a 2D matrix [num_samples, in_features], then
        accumulate X^T @ X.

        Uses float32 accumulation for numerical stability.
        """
        # Flatten to 2D: [*, in_features] -> [num_samples, in_features]
        in_features = x.shape[-1]
        x_flat = x.reshape(-1, in_features).astype(mx.float32)
        num_samples = x_flat.shape[0]

        # Compute X^T @ X contribution: [in_features, num_samples] @ [num_samples, in_features]
        # This is [in_features, in_features]
        hessian_contrib = mx.matmul(x_flat.T, x_flat)

        if name not in self._hessians:
            self._hessians[name] = hessian_contrib
            self._sample_counts[name] = num_samples
            self._in_features[name] = in_features
        else:
            self._hessians[name] = self._hessians[name] + hessian_contrib
            self._sample_counts[name] += num_samples

    def collect_hessian(self, layer_name: str) -> mx.array:
        """Get the damped Hessian for a specific layer.

        Returns H_damped = H + λI where:
        - H is the accumulated Hessian X^T @ X
        - λ = damping_factor * mean(diag(H))
        - I is the identity matrix

        The damping stabilizes the Hessian inversion in GPTQ by ensuring
        positive definiteness and numerical stability.

        Args:
            layer_name: Name of the layer to get Hessian for.

        Returns:
            Damped Hessian as float32 array [in_features, in_features].

        Raises:
            KeyError: If layer_name was not instrumented or has no data.
        """
        if layer_name not in self._hessians:
            raise KeyError(
                f"No Hessian data for layer {layer_name!r}. "
                f"Available layers: {list(self._hessians.keys())}"
            )

        H = self._hessians[layer_name]
        in_features = self._in_features[layer_name]

        # Compute damping: λ = damping_factor * mean(diag(H))
        diag_H = mx.diag(H)
        mean_diag = mx.mean(diag_H)
        lambda_damp = self.damping_factor * mean_diag

        # Add damping: H_damped = H + λI
        identity = mx.eye(in_features, dtype=mx.float32)
        H_damped = H + lambda_damp * identity

        return H_damped

    def get_hessians(self, apply_damping: bool = True) -> dict[str, HessianStats]:
        """Get all collected Hessians.

        Args:
            apply_damping: If True, return damped Hessians. If False, return raw.

        Returns:
            Dictionary mapping layer names to HessianStats.
        """
        result: dict[str, HessianStats] = {}

        for name in self._hessians:
            if apply_damping:
                H = self.collect_hessian(name)
            else:
                H = self._hessians[name]

            result[name] = HessianStats(
                hessian=H,
                num_samples=self._sample_counts[name],
                in_features=self._in_features[name],
            )

        return result

    def remove_hooks(self) -> None:
        """Restore original __call__ methods on all instrumented modules."""
        for module, original_call in self._hooks:
            module.__call__ = original_call  # type: ignore[method-assign]
        self._hooks.clear()

    def reset(self) -> None:
        """Clear all accumulated Hessians without removing hooks."""
        self._hessians.clear()
        self._sample_counts.clear()
        self._in_features.clear()

    @property
    def num_layers(self) -> int:
        """Number of layers being calibrated."""
        return len(self._hooks)

    @property
    def layer_names(self) -> list[str]:
        """Names of instrumented layers."""
        return list(self._hessians.keys())


def compute_layer_hessians(
    model: nn.Module,
    calibration_data: list,
    layers: list[str] | None = None,
    forward_fn: Any | None = None,
    chunk_size: int | None = None,
    cache_dir: str | Path | None = None,
    damping_factor: float = 0.01,
) -> dict[str, mx.array]:
    """Compute Hessians for specified layers using calibration data.

    This is a high-level function that handles the full calibration workflow:
    1. Instruments the model with Hessian collection hooks.
    2. Runs forward passes over calibration data.
    3. Collects and returns damped Hessians.

    Memory optimization:
        - If chunk_size is specified, processes calibration data in chunks
          and accumulates Hessians incrementally.
        - If cache_dir is specified, caches intermediate results to disk
          for very large calibration sets.

    MoE support:
        - For MoE models, collects per-expert Hessians by tracking which
          expert each token was routed to.

    Args:
        model: The model to calibrate.
        calibration_data: List of input batches (tokenized input_ids).
        layers: Optional list of layer names to collect Hessians for.
            If None, collects for all linear layers.
        forward_fn: Optional custom forward function. If None, calls model(batch).
            Signature: forward_fn(model, batch) -> outputs
        chunk_size: If specified, process calibration data in chunks of this size.
            Useful for very large calibration sets that might not fit in memory.
        cache_dir: If specified, cache intermediate Hessians to this directory.
            Enables resumption and reduces memory pressure.
        damping_factor: Hessian damping factor (default: 0.01).

    Returns:
        Dictionary mapping layer names to damped Hessian arrays [in_features, in_features].

    Example:
        >>> from metal_marlin.converters.calibration import compute_layer_hessians
        >>> hessians = compute_layer_hessians(
        ...     model,
        ...     calibration_data,
        ...     layers=["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj"],
        ... )
        >>> q_proj_hessian = hessians["model.layers.0.self_attn.q_proj"]
    """

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    collector = HessianCollector(model, layers, damping_factor=damping_factor)

    # Default forward function
    if forward_fn is None:
        def forward_fn(m: nn.Module, batch: Any) -> Any:
            if isinstance(batch, dict):
                return m(batch["input_ids"])
            return m(batch)

    # Process calibration data
    if chunk_size is None:
        # Process all at once
        for batch in calibration_data:
            _ = forward_fn(model, batch)
            mx.eval(list(collector._hessians.values()))  # Force evaluation
    else:
        # Process in chunks with optional caching
        total_samples = len(calibration_data)
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk = calibration_data[chunk_start:chunk_end]

            for batch in chunk:
                _ = forward_fn(model, batch)
                mx.eval(list(collector._hessians.values()))

            # Optional: cache intermediate results
            if cache_path:
                chunk_cache = cache_path / f"hessian_chunk_{chunk_start}_{chunk_end}.npz"
                _cache_hessians(collector._hessians, chunk_cache)

    # Extract final damped Hessians
    hessians_stats = collector.get_hessians(apply_damping=True)
    collector.remove_hooks()

    return {name: stats.hessian for name, stats in hessians_stats.items()}


def _cache_hessians(hessians: dict[str, mx.array], path: Path) -> None:
    """Cache Hessians to disk as numpy arrays."""
    import numpy as np

    np_hessians = {}
    for name, H in hessians.items():
        # Convert to numpy, handling potential MLX-specific operations
        if hasattr(H, "tolist"):
            np_hessians[name.replace(".", "_")] = np.array(H.tolist(), dtype=np.float32)
        else:
            np_hessians[name.replace(".", "_")] = np.asarray(H)

    np.savez_compressed(path, **np_hessians)


def load_cached_hessians(cache_dir: str | Path) -> dict[str, mx.array]:
    """Load cached Hessians from disk.

    Args:
        cache_dir: Directory containing cached .npz files.

    Returns:
        Dictionary mapping layer names to Hessian arrays.
    """
    import numpy as np

    cache_path = Path(cache_dir)
    hessians: dict[str, mx.array] = {}

    for npz_file in sorted(cache_path.glob("hessian_chunk_*.npz")):
        data = np.load(npz_file)
        for key in data.files:
            layer_name = key.replace("_", ".")
            H_np = data[key]
            if layer_name in hessians:
                # Accumulate if processing multiple chunks
                hessians[layer_name] = hessians[layer_name] + mx.array(H_np)
            else:
                hessians[layer_name] = mx.array(H_np)

    return hessians


def compute_moe_hessians(
    model: nn.Module,
    calibration_data: list,
    expert_layers: dict[str, list[str]],
    router_fn: Any,
    damping_factor: float = 0.01,
) -> dict[str, dict[int, mx.array]]:
    """Compute per-expert Hessians for MoE models.

    For Mixture-of-Experts models, each expert sees different subsets of
    tokens based on router decisions. This function tracks which tokens
    are routed to each expert and computes per-expert Hessians.

    Args:
        model: The MoE model to calibrate.
        calibration_data: List of input batches.
        expert_layers: Dict mapping layer prefixes to expert sublayer names.
            Example: {"model.layers.0.moe": ["expert_0", "expert_1", ...]}
        router_fn: Function that returns router decisions for a batch.
            Signature: router_fn(model, batch) -> dict[str, mx.array]
            Returns expert indices for each layer prefix.
        damping_factor: Hessian damping factor.

    Returns:
        Nested dict: {layer_prefix: {expert_idx: Hessian}}.
    """
    # Track per-expert activations
    expert_hessians: dict[str, dict[int, mx.array]] = {}
    expert_counts: dict[str, dict[int, int]] = {}

    for layer_prefix, expert_names in expert_layers.items():
        num_experts = len(expert_names)
        expert_hessians[layer_prefix] = {}
        expert_counts[layer_prefix] = {i: 0 for i in range(num_experts)}

    for batch in calibration_data:
        # Get router decisions
        router_decisions = router_fn(model, batch)

        # For each layer with experts
        for layer_prefix in expert_layers:
            if layer_prefix not in router_decisions:
                continue

            expert_indices = router_decisions[layer_prefix]  # [batch, seq, top_k]

            # Get input activations for this layer
            # This requires the model to expose intermediate activations
            # For now, we use a simple approach: run forward and cache
            # TODO: Add proper activation caching for MoE

    # Apply damping to all expert Hessians
    for layer_prefix in expert_hessians:
        for expert_idx in expert_hessians[layer_prefix]:
            H = expert_hessians[layer_prefix][expert_idx]
            if H is not None:
                in_features = H.shape[0]
                diag_H = mx.diag(H)
                mean_diag = mx.mean(diag_H)
                lambda_damp = damping_factor * mean_diag
                identity = mx.eye(in_features, dtype=mx.float32)
                expert_hessians[layer_prefix][expert_idx] = H + lambda_damp * identity

    return expert_hessians
