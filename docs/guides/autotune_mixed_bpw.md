# Mixed Bit-Width Kernel Auto-Tuning

This guide documents the mixed bit-width kernel auto-tuning workflow for Metal Marlin.
It is the canonical replacement for prior trellis-local autotune README variants.

## Overview

The `MixedBPWAutoTuner` class provides automatic kernel optimization for trellis-quantized models with heterogeneous bit-width quantization. It benchmarks different kernel configurations and builds optimized lookup tables for kernel selection.

## Key Features

- **Benchmark different tile sizes per bit-width**: 2-bit can use 128x128, 4-bit uses 64x64
- **Test SIMDgroup configurations**: 4, 8, 12, 16 simdgroups
- **Compare kernel variants**: Decode vs prefill vs base variants
- **Build lookup tables**: (batch_size, bit_width, hidden_dim) -> best_kernel
- **Device-specific caching**: Separate configs for M1/M2/M3/M4
- **Online adaptation**: Runtime performance feedback
- **JSON export/import**: Save and load tuned configurations

## Usage

### 1. Running Autotuning

Autotuning is typically done once per device to generate an optimized configuration file.

```python
from metal_marlin.trellis.autotune_mixed_bpw import MixedBPWAutoTuner
from metal_marlin.trellis.loader import TrellisModelLoader

# Load your model
loader = TrellisModelLoader("model_dir")

# Get a sample of layers to benchmark (you don't need all layers)
layers_to_benchmark = {
    "mlp_gate": loader.load_weight("layers.0.mlp.gate_proj"),
    "mlp_up": loader.load_weight("layers.0.mlp.up_proj"),
    "mlp_down": loader.load_weight("layers.0.mlp.down_proj"),
}

# Initialize autotuner
autotuner = MixedBPWAutoTuner(
    device_name="M4 Max",  # Auto-detected if None
    config_path="configs/autotune_mixed_bpw_m4max.json"
)

# Benchmark across different scenarios
results = autotuner.benchmark_all(
    layers=layers_to_benchmark,
    batch_sizes=[1, 4, 8, 16, 32, 64],  # Relevant batch sizes
    num_warmup=5,     # Warmup iterations
    num_iter=20,      # Benchmark iterations
    verbose=True,
)

# Export the tuned configuration
autotuner.export_config("configs/autotune_mixed_bpw_m4max.json")
```

### 2. Loading Autotuned Config at Model Startup

Load the pre-tuned configuration when initializing your model:

```python
from metal_marlin.trellis.autotune_mixed_bpw import MixedBPWAutoTuner

# Load autotuned config
autotuner = MixedBPWAutoTuner.load_config("configs/autotune_mixed_bpw_m4max.json")

# Use for kernel selection during inference
kernel_config = autotuner.select_kernel(
    batch_size=4,
    hidden_dim=4096,
    bit_widths=[2, 3, 4],  # Bit widths in this layer
    use_decode=True,       # Prefer decode variants for single-token
)
```

### 3. Online Adaptation During Inference

Record kernel latencies and adapt selection based on runtime performance:

```python
# After each kernel execution
actual_latency_ms = measure_kernel_execution_time(kernel)

# Record for adaptation
autotuner.record_latency(
    kernel_name=kernel_config.kernel_name,
    batch_size=batch_size,
    latency_ms=actual_latency_ms,
    hidden_dim=hidden_dim,
    bit_widths=layer_bit_widths,
)

# Periodically adapt selection
adapted_config = autotuner.adapt_selection(
    batch_size=current_batch_size,
    hidden_dim=current_hidden_dim,
    bit_widths=current_bit_widths,
)

if adapted_config:
    # Use adapted config if confidence is high
    kernel_config = adapted_config
```

### 4. CLI Quick Start

Run tuning without writing custom Python:

```bash
uv run python -m metal_marlin.trellis.autotune_mixed_bpw --quick --output configs/autotune_mixed_bpw.json
```

Run with explicit tuning parameters:

```bash
uv run python -m metal_marlin.trellis.autotune_mixed_bpw \
  --batch-sizes 1,4,8,16,32,64 \
  --iterations 20 \
  --warmup 5 \
  --output configs/autotune_mixed_bpw_m4max.json
```

## API Reference

### MixedBPWAutoTuner

Main auto-tuner class.

```python
autotuner = MixedBPWAutoTuner(
    device_name: Optional[str] = None,
    config_path: Optional[str | Path] = None,
)
```

**Parameters:**
- `device_name`: Device name (e.g., "M4 Max"). Auto-detected if None.
- `config_path`: Path to load/save config from/to.

### Methods

#### benchmark_layer()

Benchmark a single layer with all configurations.

```python
results = autotuner.benchmark_layer(
    layer: TrellisLinear,
    batch_size: int,
    configs: Optional[List[KernelConfig]] = None,
    num_warmup: int = 5,
    num_iter: int = 20,
) -> List[BenchmarkResult]
```

#### benchmark_all()

Benchmark all layers and batch sizes.

```python
results = autotuner.benchmark_all(
    layers: Dict[str, TrellisLinear] | List[TrellisLinear],
    batch_sizes: List[int],
    num_warmup: int = 5,
    num_iter: int = 20,
    verbose: bool = True,
) -> Dict[str, Dict[int, List[BenchmarkResult]]]
```

#### select_kernel()

Select optimal kernel from lookup table.

```python
config = autotuner.select_kernel(
    batch_size: int,
    hidden_dim: int,
    bit_widths: List[int],
    use_decode: bool = False,
    fallback_to_default: bool = True,
) -> Optional[KernelConfig]
```

**Selection Logic:**
1. Try exact match: (batch_size, hidden_dim, bit_widths)
2. Try nearest batch_size
3. Try nearest hidden_dim
4. Fallback to default config based on bit-width

#### record_latency()

Record kernel latency for online adaptation.

```python
autotuner.record_latency(
    kernel_name: str,
    batch_size: int,
    latency_ms: float,
    hidden_dim: Optional[int] = None,
    bit_widths: Optional[List[int]] = None,
)
```

#### adapt_selection()

Adapt kernel selection based on historical timings.

```python
config = autotuner.adapt_selection(
    batch_size: int,
    hidden_dim: int,
    bit_widths: List[int],
    top_k: int = 3,
) -> Optional[KernelConfig]
```

**Returns:** `KernelConfig` if confidence is high (>=5 samples), else `None`.

#### export_config()

Export autotuned configuration to JSON.

```python
json_str = autotuner.export_config(
    path: Optional[str | Path] = None,
) -> str
```

#### load_config() [classmethod]

Load autotuned configuration from JSON.

```python
autotuner = MixedBPWAutoTuner.load_config(
    path: str | Path,
) -> MixedBPWAutoTuner
```

### Data Classes

#### KernelConfig

Kernel configuration for benchmarking.

```python
@dataclass
class KernelConfig:
    kernel_name: str      # Metal kernel function name
    tile_size_m: int      # Tile size for M dimension
    tile_size_n: int      # Tile size for N dimension
    simdgroups: int       # Number of SIMDgroups
    use_fp32_acc: bool = False
    kernel_variant: str = "base"  # decode, prefill, or base
```

#### BenchmarkResult

Result from benchmarking a kernel configuration.

```python
@dataclass
class BenchmarkResult:
    config: KernelConfig
    latency_ms: float
    throughput_gbps: float
    memory_footprint_mb: float
    timestamp: float
```

## Configuration Structure

The exported JSON config has the following structure:

```json
{
  "device_name": "M4 Max",
  "device_family": "M4",
  "tile_size_mapping": {
    "2": 128,
    "3": 96,
    "4": 64,
    "8": 64
  },
  "simdgroup_mapping": {
    "2": 16,
    "3": 12,
    "4": 8,
    "8": 4
  },
  "lookup_table": {
    "4": {
      "4096": {
        "(2, 4)": {
          "kernel_name": "moe_trellis_mixed_swiglu_decode",
          "tile_size_m": 64,
          "tile_size_n": 64,
          "simdgroups": 12,
          "use_fp32_acc": false,
          "kernel_variant": "decode"
        }
      }
    }
  },
  "adaptation_enabled": true,
  "adaptation_history_len": 100,
  "benchmark_count": 250
}
```

## Device-Specific Configurations

Create separate configs for each device type:

```
configs/
├── autotune_mixed_bpw_m1_pro.json
├── autotune_m2_ultra.json
├── autotune_m3_max.json
└── autotune_mixed_bpw_m4_max.json
```

The auto-tuner auto-detects the device family and loads the appropriate config:

```python
# Auto-detect device and load appropriate config
autotuner = MixedBPWAutoTuner.load_config(
    f"configs/autotune_mixed_bpw_{autotuner.device_family.lower()}.json"
)
```

## Performance Guidelines

### Tile Size Selection

- **2-bit weights**: Larger tiles (96-160) for better memory coalescing
- **4-bit weights**: Medium tiles (48-96) balanced between memory and compute
- **8-bit weights**: Smaller tiles (32-64) due to larger memory footprint

### SIMDgroup Configuration

- **2-bit**: Higher simdgroups (12-16) for better parallelism
- **4-bit**: Medium simdgroups (8-12) balanced
- **8-bit**: Lower simdgroups (4-8) to reduce register pressure

### Kernel Variant Selection

- **Decode**: For single-token generation (batch_size=1)
- **Prefill**: For small batches (2-16 tokens)
- **Base**: For medium batches (17-32 tokens)
- **Large Batch**: For large batches (33+ tokens)

### FP32 Accumulation

Use `use_fp32_acc=True` when:
- Hidden dimension >= 1024
- High precision is required
- Numerical stability is a concern

## Testing

Run the test suite:

```bash
cd contrib/metal_marlin
uv run pytest tests/test_autotune_mixed_bpw.py -v
```

The test suite covers:
- KernelConfig and BenchmarkResult dataclasses
- AutotuneConfig initialization
- Device detection
- Config generation for single and mixed bit-widths
- Synthetic input creation
- Nearest key finding
- Kernel selection with lookup table
- Kernel selection with fallback
- Kernel selection with nearest batch/hidden_dim
- Latency recording and history trimming
- Adaptation selection with/without history
- Config export/import roundtrip
- Statistics generation

## Integration with kernel_selection_mixed.py

The auto-tuner works alongside the existing `kernel_selection_mixed.py` module:

1. **kernel_selection_mixed.py**: Runtime selection based on heuristics and expert activation patterns
2. **autotune_mixed_bpw.py**: Offline benchmarking to build optimal lookup tables

You can combine both:

```python
from metal_marlin.trellis.kernel_selection_mixed import get_mixed_kernel
from metal_marlin.trellis.autotune_mixed_bpw import MixedBPWAutoTuner

# Load autotuner for optimized base configs
autotuner = MixedBPWAutoTuner.load_config("configs/autotune_mixed_bpw_m4max.json")

# Get baseline config from autotuner
baseline_config = autotuner.select_kernel(
    batch_size=batch_size,
    hidden_dim=hidden_dim,
    bit_widths=active_expert_bits,
)

# Use kernel_selection_mixed for runtime adaptation
kernel, metadata = get_mixed_kernel(
    batch_size=batch_size,
    active_expert_bits=active_expert_bits,
    available_kernels={baseline_config.kernel_name},
    # ... other parameters
)
```

## Troubleshooting

### Import Errors

If you encounter import errors:

```python
# Ensure you're in the contrib/metal_marlin directory
import sys
sys.path.insert(0, '/path/to/contrib/metal_marlin')
```

### Benchmark Failures

If a kernel fails to benchmark:
- The auto-tuner logs warnings and continues with other configs
- Failed configs are not included in the lookup table
- Check Metal kernel compilation logs for errors

### Low Confidence Adaptation

If `adapt_selection()` always returns `None`:
- Increase the number of samples before calling it
- Reduce `top_k` parameter to consider fewer candidates
- Increase `adaptation_history_len` in the config

### Memory Issues During Benchmarking

If you run out of memory:
- Reduce `num_iter` parameter
- Benchmark fewer batch sizes at once
- Clear results: `autotuner.clear_adaptation_history()`

## Performance Tips

1. **Benchmark representative layers**: Sample different layer types (MLP, attention, MoE)
2. **Use realistic batch sizes**: Benchmark sizes you'll actually use in production
3. **Export configs per device**: M1/M2/M3/M4 have different optimal configurations
4. **Enable online adaptation**: Set `adaptation_enabled=True` for runtime optimization
5. **Periodically re-tune**: Re-run benchmarks when updating Metal kernels or models

## Future Enhancements

Potential improvements:

- Automatic detection of optimal tile sizes based on hardware capabilities
- Multi-objective optimization (latency, throughput, memory)
- A/B testing with confidence intervals
- Distributed benchmarking across multiple devices
- Integration with CI/CD for performance regression testing
