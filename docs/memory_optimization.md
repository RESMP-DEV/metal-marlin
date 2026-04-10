# Memory Optimization Guide

## Architecture Overview
- Apple Silicon: Unified Memory Architecture (UMA)
- NVIDIA CUDA: Discrete GPU with PCIe/NVLink

## Apple Silicon Zero-Copy Pipeline
- safetensors mmap → unified memory → GPU reads directly
- No pin_memory needed (shared physical RAM)
- MTLHeap sub-allocation with priority eviction

## CUDA Optimized Pipeline
- Pinned memory pool for DMA transfers
- Dedicated transfer stream for compute/copy overlap
- Stream-ordered memory allocation

## Cross-Platform Abstraction
- `device_memory.py` — platform detection and optimal transfers
- `memory_pressure.py` — proactive eviction with platform-aware queries

## Buffer Pool Architecture
- C++ MTLHeap-based pool (Metal)
- Python CUDAPinnedPool (CUDA)
- ExpertOutputRing for MoE dispatch

## Performance Characteristics
| Operation | Apple Silicon | CUDA |
|-----------|-------------|------|
| Weight load (per tensor) | 0-1 copies (UMA) | 2 copies (pin+DMA) |
| KV cache update | In-place (shared) | Device-local write |
| Expert dispatch | Ring buffer reuse | Ring buffer reuse |
| Memory pressure | System RAM monitoring | GPU VRAM monitoring |
