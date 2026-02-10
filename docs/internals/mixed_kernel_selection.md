# Intelligent Mixed-Precision Kernel Selection

## Overview

This module implements intelligent kernel selection for MoE (Mixture of Experts) layers with mixed-bit-width (BPW) quantization. The system dynamically selects optimal kernels based on:

1. **Expert Bit-Width Distribution**: Analyzes which quantization levels (2-bit, 3-bit, 4-bit) are active
2. **Batch Size**: Chooses appropriate kernel for decode, small prefill, medium prefill, or large batch
3. **Memory Pressure**: Adapts kernel selection based on available GPU memory
4. **Historical Performance**: Uses feedback loop from previous executions
5. **A/B Testing**: Balances exploration and exploitation for optimal performance

## Architecture

### Core Components

#### `kernel_selection_mixed.py`

Main selector module implementing intelligent kernel selection:

- **`MixedKernelSelector`**: Stateful selector class
- **`get_mixed_kernel()`**: Public API for kernel selection
- **`record_kernel_latency()`**: Feedback recording API
- **`get_statistics()`**: Performance statistics API

#### `kernel_selection_mixed_integration.py`

Integration examples and helper functions:

- **`MixedBpwMoEDispatcherExample`**: High-level dispatcher class
- **`get_expert_bits()`**: Extract bit-width information
- **`estimate_gpu_memory_pressure()`**: Memory pressure estimation
- **`analyze_expert_activation_pattern()`**: Expert utilization analysis

## Heuristics

### 1. Bit-Width Based Selection

The selector analyzes the distribution of active expert bit-widths:

- **>50% 2-bit experts**: Use `moe_trellis_swiglu_decode` (optimized for sparse patterns)
- **Mixed 2/3/4-bit**: Use standard `moe_trellis_swiglu` (handles mixed via dequant parameters)
- **Mostly 4-bit**: Use standard `moe_trellis_swiglu` (default)

### 2. Batch Size Based Selection

Batch size takes precedence for optimal performance:

| Batch Size | Kernel Variant | Use Case |
|------------|----------------|----------|
| 1 | `moe_trellis_swiglu_decode` | Single-token decode |
| 2-16 | `moe_trellis_swiglu_prefill4` | Small prefill (4-token SIMD) |
| 17-32 | `moe_trellis_swiglu` | Medium prefill |
| 33+ | `moe_trellis_swiglu_large_batch` | Large batch (tile_n=128) |

### 3. Specialized Decode Kernels

For common bit-width tuples, use specialized kernels with compile-time parameters:

| Bit Tuple (gate, up, down) | Kernel | Use Case |
|----------------------------|--------|----------|
| (6, 2, 3) | `moe_trellis_swiglu_decode_6_2_3` | GLM-4.7-Flash dominant |
| (6, 3, 4) | `moe_trellis_swiglu_decode_6_3_4` | Mixed 3/4-bit |
| (6, 2, 4) | `moe_trellis_swiglu_decode_6_2_4` | Mixed 2/4-bit |

### 4. Memory Pressure Adaptation

When GPU memory pressure exceeds 90% threshold:
- Falls back to base kernel to reduce temporary buffer usage
- Prioritizes stability over performance

### 5. Dynamic Feedback Loop

The system maintains a history of kernel performance:

- Stores timing data per (kernel, batch_bucket) pair
- Uses exponential moving average for trend detection
- Switches to faster kernels when statistically significant

### 6. A/B Testing

Exploration vs exploitation framework:

- **Exploration** (5% default): Randomly select alternative kernels to gather data
- **Exploitation** (95% default): Use best-known kernel from history
- **Selection Accuracy**: Tracks percentage of correct kernel choices

## Usage

### Basic Usage

```python
from metal_marlin.trellis.kernel_selection_mixed import (
    get_mixed_kernel,
    record_kernel_latency,
)

# Select kernel
kernel_name, metadata = get_mixed_kernel(
    batch_size=1,
    active_expert_bits=[2, 2, 3, 4],
    gpu_memory_pressure=0.3,
    gate_bits=6,
    up_bits=2,
    down_bits=3,
    use_fp32_acc=False,
    available_kernels=None,  # Optional: limit to available kernels
)

# Use kernel in dispatch
output = dispatch_moe_trellis_swiglu(..., kernel_name_override=kernel_name)

# Record performance feedback
record_kernel_latency(kernel_name, batch_size, latency_ms)
```

### Integration with MoE Layer

```python
from metal_marlin.trellis.kernel_selection_mixed_integration import (
    MixedBpwMoEDispatcherExample,
)

# Initialize dispatcher
dispatcher = MixedBpwMoEDispatcherExample(
    lib=metal_lib,
    expert_bit_metadata={
        0: (6, 2, 3),
        1: (6, 2, 3),
        2: (6, 3, 4),
    },
    use_fp32_acc=False,
    enable_ab_testing=True,
    exploration_rate=0.05,
)

# In forward pass
output, metadata = dispatcher.dispatch(
    activations=hidden_states,
    gate_weights=gate_weights,
    # ... other parameters ...
    expert_ids=expert_ids,
    expert_probs=expert_probs,
    # ... metadata ...
)

# Access metadata
print(f"Kernel: {metadata['kernel_name']}")
print(f"Latency: {metadata['latency_ms']:.2f}ms")
print(f"Reason: {metadata['selection_metadata']['reason']}")
```

### Performance Monitoring

```python
from metal_marlin.trellis.kernel_selection_mixed import (
    get_statistics,
    reset_history,
    set_exploration_rate,
)

# Get performance statistics
stats = get_statistics()
print(f"Selection Accuracy: {stats['selection_accuracy']:.1%}")
print(f"Total Selections: {stats['total_selections']}")
print(f"Tracked Kernels: {stats['tracked_kernels']}")

# Reset history (e.g., after model parameter update)
reset_history()

# Adjust exploration rate
set_exploration_rate(0.1)  # Increase exploration
```

## Performance Characteristics

### Expected Improvements

- **Mixed-BPW Models**: 15-20% latency improvement vs uniform selection
- **Decode Path**: Up to 17% improvement with specialized kernels
- **Large Batch**: 20-30% improvement with large_batch kernel (tile_n=128)

### Benchmarks

Run comprehensive benchmarks:

```bash
cd contrib/metal_marlin

# Run mixed-BPW kernel selection benchmarks
uv run python benchmarks/bench_moe_kernel_selection.py

# Run A/B testing framework
uv run python benchmarks/kernel_ab_test.py \
    --iterations 100 \
    --baseline moe_trellis_swiglu \
    --optimized moe_trellis_swiglu_decode

# Run targeted performance tests
uv run pytest tests/test_kernel_selection_mixed.py -v
```

## Testing

### Unit Tests

```bash
cd contrib/metal_marlin
uv run pytest tests/test_kernel_selection_mixed.py -v
```

### Integration Tests

```python
# Test with real Metal kernels
from metal_marlin.trellis.kernel_selection_mixed import MixedKernelSelector
import torch

selector = MixedKernelSelector()

# Simulate decode path with mixed bits
bits = [2, 2, 3, 4]
kernel, metadata = selector.select_kernel(batch_size=1, active_expert_bits=bits)

assert kernel in [
    "moe_trellis_swiglu_decode",
    "moe_trellis_swiglu",
]
```

### A/B Testing Framework

The system includes built-in A/B testing capabilities:

1. **Exploration**: Randomly test alternative kernels
2. **Data Collection**: Record timing for all tested kernels
3. **Statistical Analysis**: Compare means with t-test
4. **Optimization**: Switch to statistically faster kernels

See `benchmarks/kernel_ab_test.py` for details.

## Advanced Configuration

### Custom Exploration Rate

Adjust the balance between exploration and exploitation:

```python
from metal_marlin.trellis.kernel_selection_mixed import set_exploration_rate

# Conservative (always exploit)
set_exploration_rate(0.0)

# Balanced (default)
set_exploration_rate(0.05)

# Aggressive exploration
set_exploration_rate(0.2)
```

### Custom History Length

Control how much timing data to retain:

```python
selector = MixedKernelSelector(history_len=200)  # Default: 100
```

### Memory Pressure Thresholds

Customize memory pressure sensitivity:

```python
# In kernel_selection_mixed.py, modify:
if gpu_memory_pressure > 0.9:  # 90% threshold
    selected_kernel = self.KERNEL_BASE
```

## Troubleshooting

### Kernel Not Available

If a selected kernel isn't available:

```
fallback_availability: Selected kernel X not available, using Y instead
```

**Solution**: Check `available_kernels` parameter or add missing kernel to Metal library.

### High Exploration Overhead

If A/B testing is adding too much overhead:

```python
# Reduce exploration rate
set_exploration_rate(0.01)

# Or disable entirely
set_exploration_rate(0.0)
```

### Poor Selection Accuracy

If the selector isn't choosing optimal kernels:

1. Check if feedback is being recorded: `record_kernel_latency()`
2. Verify timing data is being collected: `get_statistics()`
3. Increase history length: `MixedKernelSelector(history_len=200)`
4. Reduce exploration to focus on known good kernels

## Future Enhancements

- [ ] Expert activation pattern prediction
- [ ] Learned kernel selection policies
- [ ] Multi-objective optimization (latency + memory + accuracy)
- [ ] Cross-batch kernel sharing strategies
- [ ] Dynamic kernel compilation for unseen bit-width patterns

## References

- Base kernel selection: `kernel_selection.py`
- MoE dispatch: `moe_dispatch.py`
- Metal kernels: `src/gemm_trellis.metal`
- Benchmarks: `benchmarks/bench_moe_kernel_selection.py`
