"""Vision-language projector modules for Metal Marlin.

This module provides vision-language projector implementations for multimodal
LLMs. Projectors map vision encoder outputs to the LLM's embedding space.

Supported architectures:
- LLaVA: 2-layer MLP projector
- Qwen2-VL: Perceiver resampler (cross-attention)
- InternVL: QLLaMA-style projector with learned queries

All projectors support:
- Higher precision (FP8/FP16) to preserve visual information
- Variable-length image token sequences
- Multi-image and video frame inputs

Usage:
    from metal_marlin.vision import (
        VisionProjector,
        VisionProjectorConfig,
        detect_projector_type,
    )

    # Auto-detect from HuggingFace config
    config = VisionProjectorConfig.from_hf_config(hf_config)
    projector = VisionProjector.from_config(config)

    # Forward pass
    llm_embeddings = projector(vision_features)  # [batch, num_patches, llm_hidden]
"""

from .encoder_quant import (
    VisionCalibrationConfig,
    analyze_vision_encoder_sensitivity,
    build_vision_calibration_dataset,
    build_vision_precision_map,
    build_vlm_precision_map,
    collect_vision_calibration_stats,
    pack_vlm_mixed_format,
)
from .projector import (
    InternVLProjector,
    LLaVAProjector,
    Qwen2VLProjector,
    VisionProjector,
    VisionProjectorConfig,
    detect_projector_type,
)

__all__ = [
    "VisionProjector",
    "VisionProjectorConfig",
    "LLaVAProjector",
    "Qwen2VLProjector",
    "InternVLProjector",
    "detect_projector_type",
    "VisionCalibrationConfig",
    "build_vision_calibration_dataset",
    "collect_vision_calibration_stats",
    "analyze_vision_encoder_sensitivity",
    "build_vision_precision_map",
    "build_vlm_precision_map",
    "pack_vlm_mixed_format",
]
