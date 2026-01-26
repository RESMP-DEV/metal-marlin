# Byte-Level Model Architectures

This document analyzes byte-level transformer architectures (MegaByte, ByteT5, ByT5, etc.) and their compatibility with Metal Marlin's quantization and inference infrastructure. Byte-level models are an emerging paradigm that eliminates tokenization entirely, processing raw byte sequences directly.

## Why Byte-Level Models?

Traditional transformers use subword tokenization (BPE, SentencePiece, WordPiece) to reduce sequence length. A 1000-character text becomes ~250-300 tokens. Byte-level models invert this: the same text becomes 1000 bytes (or more, for multi-byte UTF-8 characters).

### Tokenizer Limitations

| Problem | Example | Impact |
|---------|---------|--------|
| Vocabulary mismatch | " hello" ≠ "hello" (leading space) | Fragile prompts |
| Language bias | English gets 3-4 chars/token; CJK gets 1-2 | Non-English inefficiency |
| Code handling | Whitespace tokenization varies | Formatting sensitivity |
| Novel text | Rare words split arbitrarily | Poor generalization |
| Tokenizer attacks | Adversarial token boundaries | Security vulnerabilities |

### Byte-Level Advantages

1. **Universal vocabulary**: 256 possible bytes, no tokenizer training needed
2. **Language agnostic**: Same representation for all languages and scripts
3. **Robust to noise**: Character-level spelling errors don't cascade
4. **Code-native**: Consistent handling of whitespace, indentation, syntax
5. **No OOV tokens**: Any byte sequence is representable
6. **Simpler pipeline**: No tokenizer dependencies, preprocessing, or vocabulary files

### The Core Challenge

The fundamental problem: transformer attention is O(n²) in sequence length. A 4096-token context becomes a 16K-50K byte context, making naive byte-level attention 15-150× more expensive.

## MegaByte Architecture

MegaByte (Meta, 2023) addresses the sequence length problem through hierarchical attention with two levels of granularity.

### Core Concept: Global + Local Transformers

```
Input bytes: "Hello, world!" (13 bytes)
             H  e  l  l  o  ,     w  o  r  l  d  !
             │  │  │  │  │  │  │  │  │  │  │  │  │
             └──┴──┴──┴──┘  └──┴──┴──┴──┘  └──┴──┴──┘
                Patch 0       Patch 1       Patch 2
                (5 bytes)     (4 bytes)     (4 bytes)

┌──────────────────────────────────────────────────────────────────────────┐
│                         MegaByte Architecture                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input bytes [T bytes total]                                             │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Patch Embedding                                   │ │
│  │  Group consecutive bytes into patches of size P (e.g., P=4 or P=8)  │ │
│  │  Embed each byte, concatenate → [T/P patches, P * d_byte]           │ │
│  │  Linear projection → [T/P, d_global]                                │ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
│                                   │                                       │
│                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Global Transformer                                │ │
│  │  Operates on patch-level: [T/P, d_global]                           │ │
│  │  Standard self-attention between patches                             │ │
│  │  Large model (most parameters): many layers, large d_global          │ │
│  │  Output: contextualized patch embeddings [T/P, d_global]            │ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
│                                   │                                       │
│                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Local Transformer (per patch)                     │ │
│  │  For each patch independently:                                       │ │
│  │    1. Take global patch embedding as prefix                         │ │
│  │    2. Autoregressively predict bytes within patch                   │ │
│  │    3. Small model: few layers, small d_local                        │ │
│  │  Output: byte-level predictions [P bytes per patch]                 │ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
│                                   │                                       │
│                                   ▼                                       │
│  Output: next-byte logits [T, 256]                                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Computational Complexity

The hierarchical design achieves significant complexity reduction:

```
Naive byte-level transformer:
  Attention cost: O(T²) where T = number of bytes
  For 32K bytes: 32K² = 1 billion attention operations

MegaByte with patch size P:
  Global attention: O((T/P)²) = O(T²/P²)
  Local attention: O(P²) per patch, T/P patches = O(T × P)
  Total: O(T²/P² + T × P)

  For T=32K, P=8:
    Global: (32K/8)² = 4K² = 16M operations
    Local: 32K × 8 = 256K operations
    Total: ~16.3M vs 1B naive = 61× reduction
```

Optimal patch size depends on context length: larger contexts benefit from larger P.

### Architectural Parameters

| Component | Typical Configuration | Role |
|-----------|----------------------|------|
| Patch size P | 4-8 bytes | Granularity tradeoff |
| Byte embedding dim | 256-512 | Per-byte representation |
| Global d_model | 2048-4096 | Patch-level capacity |
| Global layers | 12-24 | Long-range dependencies |
| Local d_model | 256-512 | Byte-level prediction |
| Local layers | 2-4 | Within-patch modeling |
| Global heads | 16-32 | Multi-head attention |
| Local heads | 4-8 | Smaller attention |

The global transformer contains most parameters (~90%). The local transformer is intentionally small since it only needs to model short-range byte dependencies within a patch.

### Positional Encoding

MegaByte uses two levels of positional information:

```python
# Global positions: patch indices
global_pos_embed = learned_embedding[patch_idx]  # [T/P, d_global]

# Local positions: byte position within patch
local_pos_embed = learned_embedding[byte_in_patch_idx]  # [P, d_local]

# Combined during local transformer input
local_input = concat(global_output[patch_idx], byte_embed + local_pos_embed)
```

This dual encoding captures both long-range patch context and fine-grained byte ordering.

## ByteT5 and ByT5 Architectures

Google's ByT5 (2021) and subsequent ByteT5 work took a different approach: make byte-level attention efficient through architectural modifications rather than hierarchical decomposition.

### ByT5: Modified T5 for Bytes

ByT5 adapts the T5 encoder-decoder architecture:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ByT5 Architecture                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Key modifications from T5:                                          │
│                                                                       │
│  1. Vocabulary: 256 bytes + 3 special tokens (259 total)             │
│                                                                       │
│  2. Encoder: 3× more layers than decoder (e.g., 12 enc, 4 dec)       │
│     Rationale: Encoder processes longer byte sequences                │
│                                                                       │
│  3. Decoder: Fewer layers, lower overhead                            │
│     Byte-level output is still expensive                              │
│                                                                       │
│  4. No embedding sharing: Separate input/output embeddings           │
│     Byte embeddings are small (256×d), no need to share              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Efficiency Techniques

ByT5 employs several efficiency mechanisms:

**Downsampling in encoder:**
```
Input bytes: [T, d_model]
     ↓
Local attention layers (short window)
     ↓
Strided pooling (reduce by factor k=4)
     ↓
Global attention layers (reduced sequence)
     ↓
Upsampling (restore resolution)
     ↓
Output: [T, d_model]
```

**Asymmetric encoder-decoder:**
- Heavy encoder processes the long byte sequence once
- Light decoder generates output bytes autoregressively
- Cross-attention from decoder to encoder representations

## Quantization Compatibility Assessment

### Weight Quantization: Fully Compatible

Byte-level models use standard linear layers internally. All existing Metal Marlin quantization techniques apply:

| Component | Standard Approach | Compatibility |
|-----------|------------------|---------------|
| Byte embeddings | `nn.Embedding(256, d_model)` | FP16 (small, keep full precision) |
| Global QKV projections | `nn.Linear(d, 3*d)` | FP4/INT4 (standard GEMM) |
| Global FFN | `nn.Linear(d, 4*d)` | FP4/INT4 (SwiGLU compatible) |
| Local QKV projections | `nn.Linear(d, 3*d)` | FP4/INT4 (smaller, same pattern) |
| Local FFN | `nn.Linear(d, 4*d)` | FP4/INT4 |
| Output projection | `nn.Linear(d, 256)` | FP4/INT4 (very small N=256) |

**Embedding table quantization:**
The byte embedding table is only 256 × d_model elements. For d_model=4096, this is 1M parameters (2 MB in FP16). Quantization is unnecessary and may hurt quality since byte representations are foundational.

**Output head quantization:**
The output projection to 256 logits is tiny. Keep in FP16 for numerical stability in softmax.

### KV Cache Quantization: Special Considerations

The KV cache challenge is amplified for byte models due to longer sequences:

```
Token-level model (4K context):
  KV cache: 32 layers × 2(K+V) × 8 heads × 4K × 128 × 2B = 4 GB

Byte-level model (32K context, ~4K "tokens"):
  KV cache: 32 layers × 2(K+V) × 8 heads × 32K × 128 × 2B = 32 GB  ← Problem!
```

**Mitigation strategies:**

1. **FP4 KV cache**: Mandatory for byte models. Reduces to ~8 GB.

2. **Sliding window attention**: Many byte-level dependencies are local. A 4K byte window may suffice for the local transformer.

3. **MegaByte's advantage**: Global transformer's KV cache is only T/P in length. For P=8, this is 4K entries for 32K bytes, fitting comfortably in FP16.

4. **Paged attention**: Essential for variable-length byte sequences. Pre-allocating 100K bytes is wasteful.

### Hierarchical KV Cache for MegaByte

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    MegaByte KV Cache Strategy                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Global Transformer KV Cache:                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Shape: [batch, global_layers, 2, num_heads, T/P, head_dim]        │  │
│  │  For T=32K, P=8: 4K entries                                         │  │
│  │  Memory: 32 layers × 2 × 32 heads × 4K × 128 × 2B = 2 GB (FP16)    │  │
│  │  Quantized: FP4 → ~0.5 GB                                           │  │
│  │  Status: Standard approach, fully compatible                        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Local Transformer KV Cache:                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Shape: [batch, local_layers, 2, num_heads, P, head_dim]           │  │
│  │  For P=8: Only 8 entries per layer!                                 │  │
│  │  Memory: 4 layers × 2 × 8 heads × 8 × 128 × 2B = 128 KB            │  │
│  │  Status: Tiny, no quantization needed. Reset per patch.             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Key insight: The local transformer's cache is ephemeral.                 │
│  It's created fresh for each patch, processes P bytes, then discarded.   │
│  Only the global cache persists across the full sequence.                 │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### Calibration for Quantization

Byte-level models require adjusted calibration:

1. **Longer calibration sequences**: Standard 2048-token calibration translates to ~8K-16K bytes. Use longer sequences.

2. **Diverse byte patterns**: Include binary data, code, multilingual text, and special characters in calibration sets.

3. **Patch-aware grouping (MegaByte)**: The global transformer sees patch embeddings, not raw bytes. Calibration should capture patch-level activation distributions.

## Metal Kernel Requirements

### Existing Kernels: Sufficient for Most Operations

Current Metal Marlin kernels handle byte-level models without modification:

| Operation | Kernel | Byte-Level Consideration |
|-----------|--------|-------------------------|
| Global GEMM | `marlin_gemm_fp4` | Standard, works as-is |
| Local GEMM | `marlin_gemm_fp4` | Smaller matrices (d_local < d_global) |
| Global attention | `flash_attention_kv_fp4` | Longer sequences (T/P) |
| Local attention | `flash_attention` | Short sequences (P), no quantization needed |
| Embedding lookup | MLX built-in | 256-entry table, trivial |

### New Kernel Requirements

**1. Patch Embedding Kernel (Performance Optimization)**

The patch embedding operation is unique to MegaByte:

```metal
// patch_embed.metal (proposed)
kernel void patch_embed_fp4(
    device const uchar* bytes,           // [T] raw input bytes
    device const half* byte_embeddings,  // [256, d_byte] embedding table
    device half* patch_embeddings,       // [T/P, P * d_byte] output
    constant PatchParams& params,        // patch_size, d_byte, T
    uint gid [[thread_position_in_grid]]
) {
    // Each threadgroup handles one patch
    uint patch_idx = gid;
    uint byte_start = patch_idx * params.patch_size;

    // Gather P byte embeddings, concatenate to output
    for (uint i = 0; i < params.patch_size; i++) {
        uint byte_val = bytes[byte_start + i];
        // Copy embedding[byte_val] to patch_embeddings[patch_idx, i*d_byte:(i+1)*d_byte]
    }
}
```

Without this, patch embedding uses P separate embedding lookups and a concat, which is inefficient.

**2. Strided Attention Kernel (ByT5-style Downsampling)**

ByT5's strided pooling between encoder layers:

```metal
// strided_pool_attention.metal (proposed)
kernel void strided_pool(
    device const half* input,    // [T, d_model]
    device half* output,         // [T/stride, d_model]
    constant PoolParams& params, // stride, pooling type (mean/max)
    ...
) {
    // Pool 'stride' consecutive positions into one
    // Supports mean pooling (average) or max pooling
}
```

**3. Asymmetric Flash Attention (Encoder-Decoder)**

ByT5's cross-attention with asymmetric sequence lengths:

```
Q: [batch, heads, T_dec, head_dim]     (decoder, short)
K: [batch, heads, T_enc, head_dim]     (encoder, long, maybe downsampled)
V: [batch, heads, T_enc, head_dim]

Challenge: T_enc >> T_dec, need efficient cross-attention
```

Current flash attention assumes Q and KV have related lengths. Asymmetric cross-attention with very long K/V needs specialized tiling:

```metal
// flash_attention_cross_asymmetric.metal (proposed)
kernel void flash_attention_cross_asymmetric(
    device const half* Q,        // [batch, heads, T_q, head_dim]
    device const uint* K_packed, // [batch, heads, T_kv, head_dim/8] FP4
    device const uint* V_packed, // [batch, heads, T_kv, head_dim/8] FP4
    device const half* K_scales, // [batch, heads, T_kv, 1]
    device const half* V_scales,
    device half* output,
    constant CrossAttnParams& params,
    ...
) {
    // Tile K/V aggressively (T_kv can be very large)
    // Tile Q minimally (T_q is small)
    // Dequantize K/V on-the-fly
}
```

### Memory Bandwidth Considerations

Byte-level models stress memory bandwidth differently:

```
Token-level decode step:
  Read: KV cache (4 GB) + weights (7B × 0.5 = 3.5 GB) = 7.5 GB
  At 400 GB/s: ~19 ms

Byte-level decode step (MegaByte, 1 byte prediction):
  Local transformer: weights only (small, ~100 MB)
  Global transformer: Only runs every P bytes (amortized)

  Effective per-byte cost:
    Local: 100 MB / 400 GB/s = 0.25 ms
    Global (amortized, P=8): (2 GB KV + 3 GB weights) / P = 625 MB effective
                             625 MB / 400 GB/s = 1.6 ms
  Total: ~1.9 ms per byte vs ~19 ms per token

  But bytes/token ≈ 4, so:
    Byte-level: 1.9 ms × 4 = 7.6 ms per "token equivalent"
    Token-level: 19 ms per token

  Byte-level is faster per "semantic unit" due to smaller local model!
```

This is MegaByte's key insight: the local transformer is cheap because it's small.

## Implementation Roadmap

### Phase 1: Basic Support (Low Effort)

Byte-level models work today with existing infrastructure:

1. Load weights via safetensors/GGUF loaders
2. Quantize all linear layers with standard `quantize_model()`
3. Use existing flash attention kernels
4. KV cache for global transformer only

**Limitations:**
- Patch embedding is slow (Python loop)
- No specialized cross-attention for ByT5
- No strided pooling kernel

### Phase 2: MegaByte Optimization (Medium Effort)

```python
# Proposed API
from metal_marlin.architectures import MegaByteConfig, MegaByteModel

config = MegaByteConfig(
    patch_size=8,
    global_layers=12,
    global_dim=4096,
    global_heads=32,
    local_layers=4,
    local_dim=512,
    local_heads=8,
    max_bytes=65536,
)

model = MegaByteModel.from_pretrained("path/to/megabyte", config)
model = model.quantize(
    global_bits=4,    # FP4 for large global transformer
    local_bits=8,     # FP8 for small local (less aggressive)
    kv_bits=4,        # FP4 KV cache for global only
)

for byte in model.generate_stream(b"Hello, ", max_bytes=1000):
    print(chr(byte), end="")
```

New components:
- [ ] `patch_embed.metal`: Fused patch embedding kernel
- [ ] `MegaByteConfig` and `MegaByteModel` classes
- [ ] Dual KV cache management (global persistent, local ephemeral)
- [ ] Byte-level tokenizer interface (`encode()` returns bytes, `decode()` from bytes)

### Phase 3: ByT5 Support (Higher Effort)

```python
from metal_marlin.architectures import ByT5Config, ByT5Model

config = ByT5Config(
    encoder_layers=12,
    decoder_layers=4,
    d_model=1024,
    downsample_factor=4,
    max_bytes=32768,
)

model = ByT5Model.from_pretrained("google/byt5-base")
model = model.quantize(bits=4)

output_bytes = model.generate(
    input_bytes=b"translate English to German: Hello world",
    max_length=100,
)
```

New components:
- [ ] `strided_pool.metal`: Downsampling between encoder layers
- [ ] `flash_attention_cross_asymmetric.metal`: Efficient cross-attention
- [ ] `ByT5Config` and `ByT5Model` classes
- [ ] Encoder-decoder attention pattern support

### Phase 4: Advanced Optimizations (Future)

- [ ] Speculative decoding for bytes (predict multiple bytes, verify)
- [ ] Sparse byte attention (skip padding, exploit structure)
- [ ] Continuous batching for byte sequences
- [ ] Ring buffer KV cache for streaming inference

## Comparison with Alternative Approaches

### MegaByte vs Token-Level + Long Context

| Aspect | MegaByte (Byte-Level) | LLaMA-3.1 (128K Tokens) |
|--------|----------------------|-------------------------|
| Vocabulary | 256 bytes | 128K tokens |
| Input length | ~4× longer | Standard |
| Tokenizer | None | Complex SentencePiece |
| Language equity | Equal for all | English-biased |
| Code handling | Native byte representation | Token artifacts |
| Memory (weights) | Similar | Similar |
| Memory (KV cache) | Higher for naive, lower for hierarchical | Standard |
| Complexity | Hierarchical architecture | Standard transformer |
| Maturity | Research stage | Production-ready |

### When to Use Byte-Level Models

**Good fit:**
- Multilingual applications (especially non-English)
- Code generation (formatting-sensitive)
- Binary data processing
- Robustness to input noise
- Eliminating tokenizer dependencies

**Poor fit:**
- Maximum throughput needed (token-level is simpler)
- Short contexts where tokenizer overhead is negligible
- Applications with existing token-based infrastructure

## References

- [MegaByte: Predicting Million-Byte Sequences](https://arxiv.org/abs/2305.07185) (Meta, 2023)
- [ByT5: Towards a Token-Free Future](https://arxiv.org/abs/2105.13626) (Google, 2021)
- [Charformer: Fast Character Transformers](https://arxiv.org/abs/2106.12672) (Google, 2021)
- [CANINE: Pre-training an Efficient Tokenization-Free Encoder](https://arxiv.org/abs/2103.06874) (Google, 2021)
- [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852) (Parallel context processing)
