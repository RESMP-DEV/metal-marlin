"""Weight Visualization tools for analyzing quantization errors.

This module provides comprehensive visualization capabilities for:
- Comparing original vs quantized weights
- Analyzing error distributions across layers
- Visualizing quantization error heatmaps
- Generating per-layer error reports
- Comparing different quantization formats

Usage:
    from metal_marlin.visualization import WeightVisualizer

    # Create visualizer from safetensors
    viz = WeightVisualizer.from_safetensors("model.safetensors")
    
    # Compare FP4 quantization
    viz.compare_quantization("layer.0.attention.q_proj.weight", "fp4")
    
    # Generate layer-wise error report
    viz.generate_error_report(output_dir="reports/")
    
    # Plot error heatmap
    viz.plot_error_heatmap("layer.0.attention.q_proj.weight", "fp4")
"""

from __future__ import annotations

from .visualizer import WeightVisualizer
from .error_analysis import ErrorAnalyzer, ErrorMetrics
from .plots import plot_weight_comparison, plot_error_distribution, plot_layer_errors

__all__ = [
    "WeightVisualizer",
    "ErrorAnalyzer", 
    "ErrorMetrics",
    "plot_weight_comparison",
    "plot_error_distribution",
    "plot_layer_errors",
]
