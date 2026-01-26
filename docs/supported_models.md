# Supported Models

## Dense Models
| Model | Params | Compression | tok/s (M4 Max) | PPL Δ |
|-------|--------|-------------|----------------|-------|
| Qwen3-4B | 4B | 8x (FP4) | TBD | TBD |
| Qwen3-32B | 32B | 8x (FP4) | TBD | TBD |

## MoE Models
| Model | Total/Active | Experts | Compression | tok/s | PPL Δ |
|-------|--------------|---------|-------------|-------|-------|
| GLM-4.7-Flash | 47B/? | 64 (4) | 10x | TBD | TBD |
| Qwen3-30B-A3B | 30B/3B | 128 (8) | 10x | TBD | TBD |

## Hybrid Models
| Model | Architecture | Compression | tok/s | PPL Δ |
|-------|--------------|-------------|-------|-------|
| Nemotron-3-Nano | Mamba + MoE | 10x | TBD | TBD |

## Quantization Recipes

### MoE Models (FP8 + INT2)
- Attention projections: FP8
- MoE expert weights: INT2
- Router: BF16 (preserve routing)
- Expected compression: ~10x

### Dense Models (FP4)
- All weights: FP4 with group_size=128
- Embeddings: BF16
- Expected compression: ~8x
