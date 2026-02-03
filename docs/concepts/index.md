# Core Concepts

Understanding the fundamental concepts behind Metal Marlin.

## Architecture

- [**Architecture Overview**](architecture.md) — High-level system design
- [**Inference Architecture**](inference_architecture.md) — How inference works end-to-end
- [**MoE Architecture**](moe_architecture.md) — Mixture of Experts support
- [**Metal Acceleration**](metal_acceleration.md) — Apple Silicon hardware features

## Quantization

- [**Dequantization**](dequantization.md) — How weights are dequantized
- [**AWQ**](awq.md) — Activation-aware Weight Quantization
- [**Mixed Precision**](mixed_precision.md) — Per-layer precision strategies
- [**KV Cache**](kv_cache.md) — Quantized key-value cache

## Vision

- [**Vision Preprocessing**](vision_1024_implementation.md) — 1024x1024+ resolution support and ViT patches

## Sparsity

- [**Sparse Format**](sparse_format.md) — 2:4 structured sparsity support
