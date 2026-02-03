# Parakeet-TDT-0.6B Benchmark Results

**Generated**: 2026-01-29

## Performance Summary

| Configuration | Avg Real-time Factor | Avg Memory (GB) | Avg Time (s) | Samples |
|---------------|---------------------|----------------|--------------|---------|
| unknown | 0.0x | 0.000 | 0.000 | 3 |

## Audio Length Scaling Analysis

### Best Configuration by Audio Length

| Audio Length (s) | Best Config | Real-time Factor | Memory (GB) |
|------------------|-------------|------------------|-------------|
| 1.0 | hybrid_conservative | 69.1x | 0.013 |
| 5.0 | hybrid_conservative | 309.6x | 0.013 |
| 10.0 | hybrid_conservative | 575.4x | 0.013 |
| 30.0 | hybrid_conservative | 1327.6x | 0.013 |

### Performance Trends

- **fp4_baseline**: +4670.9% improvement from 1.0s to 30.0s
- **hybrid_conservative**: +1820.4% improvement from 1.0s to 30.0s
- **hybrid_aggressive**: +1868.6% improvement from 1.0s to 30.0s
- **mixed_precision**: +2331.0% improvement from 1.0s to 30.0s

## Performance Visualizations

![Throughput Comparison](charts/parakeet_throughput_comparison.png)

![Memory Usage](charts/parakeet_memory_usage.png)

## M4 Max Recommendations

### Best Overall Configuration: **hybrid_conservative**

- **Average Performance**: 570.4x real-time
- **Average Memory**: 0.013 GB
- **M4 Max Efficiency**: Excellent

### M4 Max Optimization Tips

1. **Memory Management**: M4 Max has unified memory architecture. Configurations using <100MB memory are optimal for concurrent processing.

2. **Neural Engine**: Consider configurations that leverage Apple's Neural Engine for best power efficiency.

3. **Thermal Considerations**: Sustained workloads benefit from configurations with lower peak memory usage to maintain performance.

### Audio Length Recommendations

- **1.0s audio**: Use `hybrid_conservative` (69.1x real-time)
- **5.0s audio**: Use `hybrid_conservative` (309.6x real-time)
- **10.0s audio**: Use `hybrid_conservative` (575.4x real-time)
- **30.0s audio**: Use `hybrid_conservative` (1327.6x real-time)

## Benchmark Methodology

- **Audio Sample Rate**: 16kHz
- **Test Lengths**: 1s, 5s, 10s, 30s audio segments
- **Device**: Apple Silicon (MPS backend)
- **Metrics**: Real-time factor, peak memory, transcription time
- **Configurations Tested**: FP4 baseline, hybrid conservative/aggressive, mixed precision

