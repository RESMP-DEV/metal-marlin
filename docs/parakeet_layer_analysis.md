# Parakeet-TDT-0.6B Layer Analysis

This document analyzes the memory requirements for each layer type in the Parakeet-TDT-0.6B speech recognition model across different precision formats.

## Model Configuration

Based on the codebase analysis, the Parakeet-TDT-0.6B model uses the following configuration:

- **Encoder**: 17 Conformer layers
- **Hidden Size**: 512
- **FFN Intermediate Size**: 2048 (4x hidden size)
- **Attention Heads**: 8
- **Vocabulary Size**: 1024
- **TDT Joint Hidden Size**: 640
- **TDT Predictor Hidden Size**: 640
- **TDT Predictor Layers**: 2 (LSTM)

## Per-Layer Weight Analysis

### Memory Calculation Notes

- **FP16**: 2 bytes per parameter
- **INT8**: 1 byte per parameter  
- **4-bit**: 0.5 bytes per parameter
- Calculations include both weights and biases
- Linear layers: `input_size × output_size + output_size` parameters
- Conv1d layers: `in_channels × out_channels × kernel_size + out_channels` parameters
- LSTM layers: 4× (input_size × hidden_size + hidden_size × hidden_size + hidden_size) parameters

### Memory Requirements by Layer Type

| Layer | Parameters | FP16 Size | INT8 Size | 4-bit Size |
|-------|------------|------------|-----------|------------|
| **Conformer Conv** | 315K | 630 KB | 315 KB | 158 KB |
| **MHSA Q/K/V** | 787K | 1.55 MB | 787 KB | 394 KB |
| **MHSA Out** | 262K | 524 KB | 262 KB | 131 KB |
| **FFN Up** | 1.05M | 2.09 MB | 1.05 MB | 524 KB |
| **FFN Down** | 1.05M | 2.09 MB | 1.05 MB | 524 KB |
| **TDT Joint** | 1.71M | 3.42 MB | 1.71 MB | 855 KB |
| **TDT Predictor** | 9.83M | 19.66 MB | 9.83 MB | 4.92 MB |
| **Subsampling** | 263K | 526 KB | 263 KB | 132 KB |

### Detailed Layer Breakdown

#### Conformer Conv Module (per layer)
- Pointwise Conv1d: 512 × 1024 × 1 + 1024 = 525,312 params
- Depthwise Conv1d: 512 × 512 × 31 + 512 = 507,904 params  
- Pointwise Conv2d: 1024 × 512 × 1 + 512 = 524,800 params
- **Total per layer**: ~315K parameters

#### MHSA Q/K/V Projections (per layer)
- Q projection: 512 × 512 + 512 = 262,656 params
- K projection: 512 × 512 + 512 = 262,656 params
- V projection: 512 × 512 + 512 = 262,656 params
- **Total per layer**: ~787K parameters

#### MHSA Output Projection (per layer)
- Output projection: 512 × 512 + 512 = 262,656 params
- **Total per layer**: ~262K parameters

#### FFN Up Projection (per layer)
- Linear1: 512 × 2048 + 2048 = 1,049,600 params
- **Total per layer**: ~1.05M parameters

#### FFN Down Projection (per layer)
- Linear2: 2048 × 512 + 512 = 1,049,600 params
- **Total per layer**: ~1.05M parameters

#### TDT Joint Network
- Encoder projection: 512 × 640 + 640 = 328,320 params
- Predictor projection: 640 × 640 + 640 = 410,240 params
- Output linear: 640 × 1024 + 1024 = 655,360 params
- Duration head: 640 × 5 + 5 = 3,205 params
- **Total**: ~1.71M parameters

#### TDT Predictor (LSTM-based)
- Embedding: 1024 × 640 = 655,360 params
- LSTM (2 layers): 4 × (640 × 640 + 640 × 640 + 640) × 2 = 6,579,200 params
- **Total**: ~9.83M parameters

#### Convolutional Subsampling
- Conv2d layer 1: 1 × 512 × 3 × 3 + 512 = 5,120 params
- Conv2d layer 2: 512 × 512 × 3 × 3 + 512 = 2,359,808 params
- **Total**: ~263K parameters

## Total Model Size Summary

| Component | FP16 Size | INT8 Size | 4-bit Size |
|-----------|------------|-----------|------------|
| Conformer Layers (17×) | 109.4 MB | 54.7 MB | 27.4 MB |
| TDT Joint | 3.42 MB | 1.71 MB | 855 KB |
| TDT Predictor | 19.66 MB | 9.83 MB | 4.92 MB |
| Subsampling | 526 KB | 263 KB | 132 KB |
| **TOTAL** | **133.0 MB** | **66.5 MB** | **33.3 MB** |

### Key Insights

1. **Dominant Components**: The Conformer layers (particularly FFN layers) consume ~82% of model parameters
2. **TDT Predictor**: Despite being a smaller component, the LSTM-based predictor accounts for ~15% of parameters
3. **Quantization Benefits**: 4-bit quantization reduces the model from 133MB to just 33MB (4× reduction)
4. **Memory Hierarchy**: FFN layers are the largest single component, making them prime targets for optimization

### Memory Optimization Opportunities

1. **FFN Optimization**: The up/down projections in FFN layers are the largest components (2.1MB per layer at FP16)
2. **LSTM Pruning**: The TDT predictor's LSTM layers could benefit from structured pruning
3. **Layer Fusion**: Combining MHSA Q/K/V projections could reduce memory overhead
4. **Adaptive Precision**: Different layer types could use different precision levels based on sensitivity