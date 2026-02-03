# Compatibility Matrix

Hardware and software compatibility for Metal Marlin across Apple Silicon generations.

## Apple Silicon Support

| Generation | Models | Metal Version | Status | Notes |
|------------|--------|---------------|--------|-------|
| **M1** | M1, M1 Pro, M1 Max, M1 Ultra | Metal 3 | ✅ Supported | Full support for all quantization modes |
| **M2** | M2, M2 Pro, M2 Max, M2 Ultra | Metal 3 | ✅ Supported | Enhanced performance with hardware ray tracing |
| **M3** | M3, M3 Pro, M3 Max | Metal 3 | ✅ Supported | Dynamic caching improves kernel dispatch |
| **M4** | M4, M4 Pro, M4 Max | Metal 3 | ✅ Supported | Optimized for tensor cores and mesh shaders |

## macOS Version Requirements

| macOS Version | Minimum Chip | Metal Features | Status |
|---------------|--------------|----------------|--------|
| **Ventura 13.0+** | M1 | Metal 3 baseline | ✅ Minimum required |
| **Sonoma 14.0+** | M1 | Improved shader compiler | ✅ Recommended |
| **Sequoia 15.0+** | M1 | Dynamic caching, ray tracing | ✅ Best performance |

## Quantization Format Support

| Format | M1 | M2 | M3 | M4 | Notes |
|--------|----|----|----|----|-------|
| **FP8** | ✅ | ✅ | ✅ | ✅ | IEEE 754 E4M3/E5M2 |
| **FP4** | ✅ | ✅ | ✅ | ✅ | Custom 4-bit float |
| **INT4** | ✅ | ✅ | ✅ | ✅ | GPTQ/AWQ compatible |
| **INT3** | ✅ | ✅ | ✅ | ✅ | Requires group_size ≤ 128 |
| **INT2** | ✅ | ✅ | ✅ | ✅ | Experimental, high compression |

## Performance Characteristics

### Compute Units

| Chip | GPU Cores | Unified Memory Bandwidth | Peak TFLOPS (FP16) |
|------|-----------|--------------------------|---------------------|
| **M1** | 8 | 68 GB/s | 2.6 |
| **M1 Pro** | 16 | 200 GB/s | 5.2 |
| **M1 Max** | 32 | 400 GB/s | 10.4 |
| **M1 Ultra** | 64 | 800 GB/s | 20.8 |
| **M2** | 10 | 100 GB/s | 3.6 |
| **M2 Pro** | 19 | 200 GB/s | 6.8 |
| **M2 Max** | 38 | 400 GB/s | 13.6 |
| **M2 Ultra** | 76 | 800 GB/s | 27.2 |
| **M3** | 10 | 100 GB/s | 4.0 |
| **M3 Pro** | 18 | 150 GB/s | 7.2 |
| **M3 Max** | 40 | 400 GB/s | 14.0 |
| **M4** | 10 | 120 GB/s | 4.5 |
| **M4 Pro** | 20 | 273 GB/s | 8.0 |
| **M4 Max** | 40 | 546 GB/s | 16.0 |

### Memory Limits

| Chip | Max Unified Memory | Recommended Model Size (INT4) |
|------|--------------------|-------------------------------|
| **M1/M2** | 16 GB | 7B parameters |
| **M1/M2 Pro** | 32 GB | 13B parameters |
| **M1/M2 Max** | 64 GB | 34B parameters |
| **M1/M2 Ultra** | 192 GB | 70B+ parameters |
| **M3** | 24 GB | 13B parameters |
| **M3 Pro** | 36 GB | 22B parameters |
| **M3 Max** | 128 GB | 70B parameters |
| **M4** | 32 GB | 13B parameters |
| **M4 Pro** | 64 GB | 34B parameters |
| **M4 Max** | 128 GB | 70B parameters |

## Feature Support

### MoE (Mixture of Experts)

| Chip | Max Experts | Expert Parallelism | Status |
|------|-------------|-------------------|--------|
| **M1** | 64 | ✅ | Supported |
| **M1 Pro+** | 256 | ✅ | Supported |
| **M2+** | 256 | ✅ | Enhanced dispatch |
| **M3+** | 256 | ✅ | Dynamic caching |
| **M4+** | 256 | ✅ | Optimized routing |

### Group Size Support

| Group Size | M1 | M2 | M3 | M4 | Performance Impact |
|------------|----|----|----|----|-------------------|
| **32** | ✅ | ✅ | ✅ | ✅ | Highest precision |
| **64** | ✅ | ✅ | ✅ | ✅ | Balanced |
| **128** | ✅ | ✅ | ✅ | ✅ | Recommended default |
| **256** | ✅ | ✅ | ✅ | ✅ | Fastest inference |

## Tested Models

| Model Family | Size Range | Tested Chips | Status |
|--------------|------------|--------------|--------|
| **Qwen** | 0.5B - 72B | M1, M2, M3, M4 | ✅ Production |
| **LLaMA** | 7B - 70B | M1, M2, M3, M4 | ✅ Production |
| **Mistral** | 7B - 22B | M1, M2, M3 | ✅ Production |
| **DeepSeek** | 1.5B - 236B (MoE) | M1 Ultra, M2 Ultra, M4 Max | ✅ MoE validated |
| **GLM** | 4B - 9B | M1, M2, M3, M4 | ✅ Production |

## Known Limitations

### M1 Generation
- **Issue**: No hardware ray tracing
- **Impact**: Slightly slower kernel dispatch vs M2+
- **Workaround**: Use Metal 3 baseline features only

### Memory-Constrained Systems
- **Issue**: Models >70B require 128GB+ RAM
- **Impact**: Limited to M3/M4 Max or M1/M2 Ultra
- **Workaround**: Use aggressive quantization (INT2/INT3)

### macOS < 13.0
- **Issue**: Metal 3 APIs unavailable
- **Impact**: Cannot run Metal Marlin
- **Workaround**: Upgrade to macOS Ventura or later

## Checking Your System

```bash
# Check macOS version
sw_vers

# Check chip model
sysctl -n machdep.cpu.brand_string

# Check Metal feature set
system_profiler SPDisplaysDataType | grep "Metal"

# Check unified memory
sysctl hw.memsize
```

## Recommended Configurations

### Development
- **Minimum**: M1 with 16GB, macOS 13.0+
- **Recommended**: M2 Pro with 32GB, macOS 14.0+

### Production Inference
- **Small models (≤7B)**: M1/M2 with 16GB
- **Medium models (7B-34B)**: M1/M2 Pro/Max with 32-64GB
- **Large models (34B-70B)**: M1/M2 Ultra or M3/M4 Max with 128GB
- **MoE models**: M2 Ultra or M4 Max for optimal performance

## Future Support

### Planned
- **M5 generation**: Expected full compatibility
- **macOS 16.0+**: Leverage new Metal 4 features
- **visionOS**: Experimental support for Apple Vision Pro

### Under Investigation
- **Cross-device dispatch**: Multi-Mac inference clustering
- **External GPU**: Thunderbolt eGPU support (limited by Metal architecture)

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for chip-specific issues.

For benchmark results across generations, see [performance.md](performance.md).
