"""Paged KV cache: vLLM-style block management for variable-length sequences.

Standard KV cache components:
- BlockAllocator, KVBlock, KVBlockConfig: Block-level storage management
- PageTable, SequenceState: Sequence-to-block mapping
- paged_attention, paged_attention_v1: Attention over paged cache

VLM (Vision-Language Models) extensions:
- MultimodalBlockAllocator: Tracks image vs text token boundaries
- Prefix caching for repeated image contexts (same image, different questions)
- VisionEncoderCache: Caches vision encoder outputs to skip redundant forward passes

MLA (Multi-Head Latent Attention) cache components:
- MLACacheConfig, MLABlock, MLABlockAllocator: Compressed latent storage
- mla_attention: Attention with on-demand decompression
- compare_memory_usage: Memory comparison utilities

MLA provides ~8x memory reduction by storing compressed latents instead of
full K, V tensors. See metal_marlin.paged.mla_cache for details.
"""

from .allocator import (
    BlockAllocator,
    BlockState,
    ImageRegion,
    MultimodalBlockAllocator,
    MultimodalBlockState,
    SequenceModality,
    TokenModality,
    VisionEncoderCache,
)
from .attention import paged_attention, paged_attention_v1, write_kv_to_blocks
from .kv_block import KVBlock, KVBlockConfig
from .mla_cache import (
    MLABlock,
    MLABlockAllocator,
    MLABlockState,
    MLACacheConfig,
    compare_memory_usage,
    mla_attention,
)
from .page_table import PageTable, SequenceState

__all__ = [
    # Core allocator
    "BlockAllocator",
    "BlockState",
    # Multimodal extensions
    "ImageRegion",
    "MultimodalBlockAllocator",
    "MultimodalBlockState",
    "SequenceModality",
    "TokenModality",
    "VisionEncoderCache",
    # KV block management
    "KVBlock",
    "KVBlockConfig",
    "PageTable",
    "SequenceState",
    # Attention operations
    "paged_attention",
    "paged_attention_v1",
    "write_kv_to_blocks",
    # MLA cache
    "MLABlock",
    "MLABlockAllocator",
    "MLABlockState",
    "MLACacheConfig",
    "compare_memory_usage",
    "mla_attention",
]
