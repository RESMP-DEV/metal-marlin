"""Error analysis utilities for quantization visualization.

Provides structured error metrics and statistical analysis for
quantized weight tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class ErrorMetrics:
    """Comprehensive error metrics for quantization analysis.
    
    Attributes:
        mse: Mean Squared Error between original and quantized
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        max_error: Maximum absolute error
        mean_relative_error: Mean relative error (|error| / |original|)
        median_relative_error: Median relative error
        psnr: Peak Signal-to-Noise Ratio in dB
        snr: Signal-to-Noise Ratio in dB
        cosine_similarity: Cosine similarity between flattened tensors
        correlation: Pearson correlation coefficient
        bits_per_weight: Effective bits per weight after quantization
        compression_ratio: Compression ratio achieved
        outlier_ratio: Fraction of weights with >10% relative error
        quantization_skew: Asymmetry in error distribution
        quantization_kurtosis: Tail heaviness of error distribution
    """
    mse: float
    rmse: float
    mae: float
    max_error: float
    mean_relative_error: float
    median_relative_error: float
    psnr: float
    snr: float
    cosine_similarity: float
    correlation: float
    bits_per_weight: float
    compression_ratio: float
    outlier_ratio: float
    quantization_skew: float
    quantization_kurtosis: float
    
    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_error": self.max_error,
            "mean_relative_error": self.mean_relative_error,
            "median_relative_error": self.median_relative_error,
            "psnr": self.psnr,
            "snr": self.snr,
            "cosine_similarity": self.cosine_similarity,
            "correlation": self.correlation,
            "bits_per_weight": self.bits_per_weight,
            "compression_ratio": self.compression_ratio,
            "outlier_ratio": self.outlier_ratio,
            "quantization_skew": self.quantization_skew,
            "quantization_kurtosis": self.quantization_kurtosis,
        }
    
    @classmethod
    def from_tensors(
        cls,
        original: npt.NDArray[np.float32],
        quantized: npt.NDArray[np.float32],
        bits_per_weight: float = 4.0,
    ) -> ErrorMetrics:
        """Compute error metrics from original and quantized tensors.
        
        Args:
            original: Original weight tensor
            quantized: Quantized/dequantized weight tensor
            bits_per_weight: Bits per weight for compression ratio calculation
            
        Returns:
            ErrorMetrics instance with computed values
        """
        orig_f = original.astype(np.float32).flatten()
        quant_f = quantized.astype(np.float32).flatten()
        
        # Basic error metrics
        diff = orig_f - quant_f
        abs_diff = np.abs(diff)
        
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_diff))
        max_error = float(np.max(abs_diff))
        
        # Relative errors
        rel_err = abs_diff / (np.abs(orig_f) + 1e-10)
        mean_relative_error = float(np.mean(rel_err))
        median_relative_error = float(np.median(rel_err))
        
        # Signal quality metrics
        signal_power = np.mean(orig_f ** 2)
        noise_power = mse
        snr = float(10 * np.log10(signal_power / (noise_power + 1e-10)))
        
        max_val = np.max(np.abs(orig_f))
        psnr = float(20 * np.log10(max_val / (rmse + 1e-10)))
        
        # Similarity metrics
        cosine_sim = float(
            np.dot(orig_f, quant_f) / (np.linalg.norm(orig_f) * np.linalg.norm(quant_f) + 1e-10)
        )
        
        # Pearson correlation
        if len(orig_f) > 1:
            correlation = float(np.corrcoef(orig_f, quant_f)[0, 1])
        else:
            correlation = 1.0
        
        # Outlier ratio (>10% relative error)
        outlier_ratio = float(np.mean(rel_err > 0.1))
        
        # Distribution shape
        if len(diff) > 3:
            quantization_skew = float(np.mean((diff - np.mean(diff)) ** 3) / (np.std(diff) ** 3 + 1e-10))
            quantization_kurtosis = float(np.mean((diff - np.mean(diff)) ** 4) / (np.std(diff) ** 4 + 1e-10) - 3)
        else:
            quantization_skew = 0.0
            quantization_kurtosis = 0.0
        
        # Compression (assuming original is FP16 = 16 bits)
        compression_ratio = 16.0 / bits_per_weight
        
        return cls(
            mse=mse,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            mean_relative_error=mean_relative_error,
            median_relative_error=median_relative_error,
            psnr=psnr,
            snr=snr,
            cosine_similarity=cosine_sim,
            correlation=correlation,
            bits_per_weight=bits_per_weight,
            compression_ratio=compression_ratio,
            outlier_ratio=outlier_ratio,
            quantization_skew=quantization_skew,
            quantization_kurtosis=quantization_kurtosis,
        )


class ErrorAnalyzer:
    """Analyzer for quantization errors across multiple layers and formats."""
    
    def __init__(self) -> None:
        """Initialize the error analyzer."""
        self.layer_metrics: dict[str, dict[str, ErrorMetrics]] = {}
        self.format_summaries: dict[str, dict[str, float]] = {}
    
    def add_layer_metrics(
        self,
        layer_name: str,
        format_name: str,
        metrics: ErrorMetrics,
    ) -> None:
        """Add metrics for a layer.
        
        Args:
            layer_name: Name of the layer/tensor
            format_name: Quantization format (e.g., "fp4", "int8")
            metrics: ErrorMetrics instance
        """
        if layer_name not in self.layer_metrics:
            self.layer_metrics[layer_name] = {}
        self.layer_metrics[layer_name][format_name] = metrics
    
    def get_layer_comparison(
        self,
        layer_name: str,
    ) -> dict[str, ErrorMetrics] | None:
        """Get metrics for all formats of a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dict mapping format names to metrics, or None if layer not found
        """
        return self.layer_metrics.get(layer_name)
    
    def get_format_summary(self, format_name: str) -> dict[str, float]:
        """Compute aggregate statistics for a quantization format.
        
        Args:
            format_name: Quantization format to summarize
            
        Returns:
            Dict with aggregate statistics across all layers
        """
        metrics_list = [
            layer_metrics[format_name]
            for layer_metrics in self.layer_metrics.values()
            if format_name in layer_metrics
        ]
        
        if not metrics_list:
            return {}
        
        return {
            "mean_rmse": float(np.mean([m.rmse for m in metrics_list])),
            "max_rmse": float(np.max([m.rmse for m in metrics_list])),
            "mean_mae": float(np.mean([m.mae for m in metrics_list])),
            "mean_snr": float(np.mean([m.snr for m in metrics_list])),
            "min_snr": float(np.min([m.snr for m in metrics_list])),
            "mean_cosine": float(np.mean([m.cosine_similarity for m in metrics_list])),
            "mean_outlier_ratio": float(np.mean([m.outlier_ratio for m in metrics_list])),
            "layer_count": len(metrics_list),
        }
    
    def get_best_format_per_layer(self, metric: str = "rmse") -> dict[str, tuple[str, float]]:
        """Find the best quantization format for each layer.
        
        Args:
            metric: Metric to optimize ("rmse", "mae", "snr", etc.)
            
        Returns:
            Dict mapping layer names to (best_format, metric_value)
        """
        best_formats = {}
        
        for layer_name, formats in self.layer_metrics.items():
            if not formats:
                continue
                
            best_format = None
            best_value = float("inf")
            
            for fmt, metrics in formats.items():
                value = getattr(metrics, metric, float("inf"))
                # For SNR and similarity metrics, higher is better
                if metric in ("snr", "psnr", "cosine_similarity", "correlation"):
                    value = -value  # Negate for min comparison
                    
                if value < best_value:
                    best_value = value
                    best_format = fmt
            
            if best_format:
                actual_value = getattr(formats[best_format], metric)
                best_formats[layer_name] = (best_format, actual_value)
        
        return best_formats
    
    def identify_problematic_layers(
        self,
        format_name: str,
        threshold_rmse: float | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Identify layers with highest quantization error.
        
        Args:
            format_name: Quantization format to analyze
            threshold_rmse: Optional RMSE threshold for filtering
            top_k: Number of top problematic layers to return
            
        Returns:
            List of (layer_name, rmse) tuples sorted by RMSE descending
        """
        layer_rmse = []
        
        for layer_name, formats in self.layer_metrics.items():
            if format_name in formats:
                rmse = formats[format_name].rmse
                if threshold_rmse is None or rmse > threshold_rmse:
                    layer_rmse.append((layer_name, rmse))
        
        layer_rmse.sort(key=lambda x: x[1], reverse=True)
        return layer_rmse[:top_k]
