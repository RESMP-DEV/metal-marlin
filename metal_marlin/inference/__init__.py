"""Optimized inference pipelines for dense transformer models.

This subpackage provides specialized routines for both phases of inference:

PREFILL (Prompt Processing):
- Large batch of tokens (full prompt)
- Compute-bound (large GEMM operations)
- KV cache write-heavy
- Target: >10K tok/s throughput on M4 Max

DECODE (Token Generation):
- M=1 (single token at a time)
- Memory-bound (small GEMM, large KV read)
- Latency-critical (impacts tok/s directly)
- Target: <5ms per token latency on M4 Max for 30B models

Usage:
    from metal_marlin.inference import (
        # Prefill
        chunked_prefill,
        parallel_kv_write,
        flash_prefill_attention,
        PrefillConfig,
        PrefillStats,
        # Decode
        persistent_decode_step,
        quantized_kv_attention,
        DecodeConfig,
        DecodeState,
    )

    # Chunked prefill for long prompts
    config = PrefillConfig(chunk_size=2048, use_flash_attention=True)
    logits, stats = chunked_prefill(model, input_ids, kv_cache, config)
    print(f"Prefill: {stats.tokens_per_second:.0f} tok/s")

    # Decode phase
    decode_config = DecodeConfig(
        num_layers=32,
        hidden_size=4096,
        kv_cache_dtype="fp8",
    )
    state = DecodeState(decode_config, model_weights)
    hidden, new_k, new_v = persistent_decode_step(
        hidden_states=h,
        state=state,
        position=seq_len,
    )
"""

from __future__ import annotations

from .decode import (
    DecodeConfig,
    DecodePerfStats,
    DecodeState,
    LayerWeights,
    fused_qkv_projection,
    persistent_decode_step,
    quantized_kv_attention,
    select_decode_kernel,
)
from .mmfp4_pipeline import MMFP4Pipeline, StreamingOutput
from .pipeline import (
    GenerationConfig,
    MarlinModel,
    MarlinPipeline,
    MetalMarlinModel,
    ModelConfig,
    ModelInfo,
    chat,
    dequantize_fp4_torch,
    get_device,
    load_quantized_model,
    load_safetensors_torch,
)
from .pipeline_v2 import TransformersMarlinPipeline
from .prefill import (
    BatchedKVResult,
    PrefillConfig,
    PrefillModel,
    PrefillStats,
    SpeculativePrefillConfig,
    batched_kv_projection,
    chunked_prefill,
    flash_prefill_attention,
    parallel_kv_write,
    speculative_prefill,
)

__all__ = [
    # Decode exports
    "DecodeConfig",
    "DecodePerfStats",
    "DecodeState",
    "LayerWeights",
    "fused_qkv_projection",
    "persistent_decode_step",
    "quantized_kv_attention",
    "select_decode_kernel",
    # Pipeline exports (high-level API)
    "GenerationConfig",
    "MMFP4Pipeline",
    "StreamingOutput",
    "MarlinModel",
    "MarlinPipeline",
    "MetalMarlinModel",
    "ModelConfig",
    "ModelInfo",
    "TransformersMarlinPipeline",
    "chat",
    "dequantize_fp4_torch",
    "get_device",
    "load_quantized_model",
    "load_safetensors_torch",
    # Prefill exports
    "BatchedKVResult",
    "PrefillConfig",
    "PrefillModel",
    "PrefillStats",
    "SpeculativePrefillConfig",
    "batched_kv_projection",
    "chunked_prefill",
    "flash_prefill_attention",
    "parallel_kv_write",
    "speculative_prefill",
]
