# Trellis v3 Format Specification

Trellis v3 is a flat shard format for storing quantized neural network weights, compatible with the HuggingFace ecosystem while preserving Trellis-specific quantization metadata.

## Overview

Trellis v3 replaces the hierarchical layer-based directory structure of v2 with a flat shard layout similar to HuggingFace's safetensors format. This improves:

- **Compatibility**: Native support with HuggingFace `from_pretrained()` loaders
- **Streaming**: Sequential reads without directory traversal
- **Storage efficiency**: Reduced filesystem overhead for many small files
- **Tooling**: Standard tools work without custom plugins

## Format Comparison

### Trellis v2 (Legacy)
```
models/Model-Trellis-3bpw/
├── layer_0000/
│   ├── index.json
│   └── tensor_0000.safetensors
├── layer_0001/
│   └── ...
├── base_weights.safetensors
├── quantization_index.json
└── config.json
```

### Trellis v3 (New)
```
models/Model-Trellis-3bpw/
├── model-00001-of-00010.safetensors  # layers 0-4
├── model-00002-of-00010.safetensors  # layers 5-9
├── model.safetensors.index.json      # weight_map
├── quantization_config.json          # quant metadata
└── config.json
```

## File Specifications

### 1. Shard Naming Convention

Shards follow the HuggingFace standard naming pattern:

```
model-{shard:05d}-of-{total:05d}.safetensors
```

Examples:
- `model-00001-of-00010.safetensors` - First shard of 10
- `model-00005-of-00012.safetensors` - Fifth shard of 12

### 2. Shard Size Target

Default shard size: **2GB per shard**

This can be configured during quantization:

```python
# quantization script
quantizer.save(
    output_path="models/MyModel-Trellis-3bpw",
    shard_size="2GB",  # Options: "1GB", "2GB", "5GB", "10GB", or bytes
)
```

Shard size guidelines:
- **1GB**: Better for slow connections, more HTTP requests
- **2GB** (default): Good balance for most use cases
- **5GB**: Faster for high-bandwidth connections
- **10GB**: Maximum for single-file models

### 3. Tensor Naming

Trellis v3 uses dot-separated component notation (HuggingFace style):

```
{original_name}.{component}
```

Example tensor names:
```
model.layers.0.mlp.down_proj.weight.indices
model.layers.0.mlp.down_proj.weight.scales
model.layers.0.mlp.down_proj.weight.su
model.layers.0.mlp.down_proj.weight.sv
```

### 4. Components Per Weight

Each quantized weight consists of 4 components:

| Component | Tensor Suffix | Shape | Dtype | Description |
|-----------|---------------|-------|-------|-------------|
| Indices | `.indices` | `[tiles_k, tiles_n, packed_size]` | `uint8` | Bit-packed trellis codebook indices |
| Scales | `.scales` | `[n_groups, N]` | `float32` | Per-tile or per-group scaling factors |
| Row Signs | `.su` | `[K]` | `float32` | Row sign vector for Hadamard restoration |
| Column Signs | `.sv` | `[N]` | `float32` | Column sign vector for Hadamard restoration |

Where:
- `K`: Input dimension (rows)
- `N`: Output dimension (columns)
- `tiles_k = ceil(K / 16)`, `tiles_n = ceil(N / 16)`: Number of 16x16 tiles
- `n_groups`: Number of scale groups (typically `tiles_k`)
- `packed_size = ceil(256 * bits / 8)`: Bytes per tile for packed indices

### 5. model.safetensors.index.json

The index file maps tensor names to their containing shards:

```json
{
  "metadata": {
    "total_size": 15854452736,
    "format": "trellis_v3",
    " quantization": {
      "bits_per_weight": 3.0,
      "method": "trellis_ldlq",
      "hadamard_transform": true
    }
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00010.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.gate_proj.weight.indices": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.gate_proj.weight.scales": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.gate_proj.weight.su": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.gate_proj.weight.sv": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.up_proj.weight.indices": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.up_proj.weight.scales": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.up_proj.weight.su": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.up_proj.weight.sv": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.down_proj.weight.indices": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.down_proj.weight.scales": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.down_proj.weight.su": "model-00001-of-00010.safetensors",
    "model.layers.0.mlp.down_proj.weight.sv": "model-00001-of-00010.safetensors",
    "model.layers.31.mlp.down_proj.weight.sv": "model-00010-of-00010.safetensors",
    "model.norm.weight": "model-00010-of-00010.safetensors",
    "lm_head.weight": "model-00010-of-00010.safetensors"
  }
}
```

### 6. quantization_config.json

Contains per-tensor quantization metadata:

```json
{
  "quantization_version": "trellis_v3",
  "quantization_method": "trellis_ldlq",
  "global_config": {
    "average_bits_per_weight": 3.0,
    "target_bits": 3,
    "hadamard_transform": true,
    "tile_size": 16,
    "codebook_size": 256,
    "scale_groups": "per_tile"
  },
  "tensor_metadata": {
    "model.layers.0.mlp.gate_proj.weight": {
      "bits": 3,
      "shape": [4096, 14336],
      "mse": 0.00042,
      "original_bytes": 234881024,
      "compressed_bytes": 11010048,
      "compression_ratio": 21.33
    },
    "model.layers.0.mlp.up_proj.weight": {
      "bits": 3,
      "shape": [4096, 14336],
      "mse": 0.00038,
      "original_bytes": 234881024,
      "compressed_bytes": 11010048,
      "compression_ratio": 21.33
    },
    "model.layers.0.mlp.down_proj.weight": {
      "bits": 4,
      "shape": [14336, 4096],
      "mse": 0.00051,
      "original_bytes": 234881024,
      "compressed_bytes": 14680064,
      "compression_ratio": 16.00
    }
  },
  "layer_allocation": {
    "0": {"mlp.gate_proj": 3, "mlp.up_proj": 3, "mlp.down_proj": 4},
    "1": {"mlp.gate_proj": 3, "mlp.up_proj": 3, "mlp.down_proj": 3},
    "...": {}
  }
}
```

Field descriptions:

| Field | Type | Description |
|-------|------|-------------|
| `quantization_version` | string | Format version: `"trellis_v3"` |
| `quantization_method` | string | Algorithm: `"trellis_ldlq"`, `"trellis_viterbi"` |
| `global_config.average_bits_per_weight` | float | Average across all tensors |
| `global_config.target_bits` | int | Target bit width (2-8) |
| `global_config.hadamard_transform` | bool | Whether Hadamard rotation was applied |
| `global_config.tile_size` | int | Quantization tile size (always 16) |
| `tensor_metadata.{name}.bits` | int | Bits allocated to this tensor |
| `tensor_metadata.{name}.shape` | [int, int] | Original weight shape [K, N] |
| `tensor_metadata.{name}.mse` | float | Quantization mean squared error |
| `tensor_metadata.{name}.compression_ratio` | float | Original size / compressed size |

## Packed Index Format

Trellis v3 stores indices in packed uint8 format (same as v2):

```
[1 byte header] + [packed data]
```

Header byte: Bits value (2-8)

Packed data layout:
- Each tile contains 256 indices (16x16)
- Indices are packed bit-contiguously
- 2-bit: 4 indices per byte
- 3-bit: 8 indices per 3 bytes
- 4-bit: 2 indices per byte
- 5-bit: 8 indices per 5 bytes
- 6-bit: 4 indices per 3 bytes
- 8-bit: 1 index per byte

Example for 3-bit quantization:
```python
# Original: [256] int16 indices per tile
# Packed: 1 + ceil(256 * 3 / 8) = 97 bytes per tile
# Header: 0x03 (3 bits)
# Data: 96 bytes containing bit-packed indices
```

## Migration from v2 to v3

### Automatic Migration

Use the built-in migration tool:

```bash
python -m metal_marlin.tools.migrate_v2_to_v3 \
    --input models/Model-Trellis-v2 \
    --output models/Model-Trellis-v3 \
    --shard-size 2GB
```

### Manual Migration

```python
from pathlib import Path
from metal_marlin.trellis.loader import TrellisModelLoader
from metal_marlin.trellis.converter import convert_to_v3

# Load v2 model
loader = TrellisModelLoader("models/Model-Trellis-v2")

# Convert and save as v3
convert_to_v3(
    loader,
    output_path="models/Model-Trellis-v3",
    shard_size="2GB",  # or bytes: 2 * 1024**3
)
```

### Migration Steps

1. **Load v2 structure**: Read layer directories and index files
2. **Collect tensors**: Gather all weight components
3. **Rename tensors**: Convert `__` separator to `.` notation
   - `model__layers__0__mlp__gate_proj__weight__indices` → `model.layers.0.mlp.gate_proj.weight.indices`
4. **Build weight map**: Assign tensors to shards based on size targets
5. **Write shards**: Save safetensors files with HuggingFile-compatible naming
6. **Generate index**: Create `model.safetensors.index.json`
7. **Migrate metadata**: Convert layer `index.json` files to `quantization_config.json`

### Compatibility

| Feature | v2 | v3 |
|---------|----|----|
| HuggingFace Hub upload | ❌ | ✅ |
| `transformers` loader | ❌ | ✅ (with adapter) |
| Metal Marlin loader | ✅ | ✅ |
| Sequential streaming | ❌ | ✅ |
| Random layer access | ✅ | ✅ (via index) |

## Loading v3 Models

### With Metal Marlin

```python
from metal_marlin.trellis.loader import TrellisModelLoader

# Auto-detects v3 format
loader = TrellisModelLoader("models/Model-Trellis-v3")

# Load specific layer
weights = loader.load_layer(0)
```

### With HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM
from metal_marlin.hf_adapter import load_trellis_model

# Use adapter for native HF loading
model = load_trellis_model(
    "models/Model-Trellis-v3",
    torch_dtype="auto",
    device_map="auto",
)
```

### Direct Safetensors Access

```python
from safetensors import safe_open

# Read index
import json
with open("models/Model-Trellis-v3/model.safetensors.index.json") as f:
    index = json.load(f)

# Load specific tensor from appropriate shard
tensor_name = "model.layers.0.mlp.gate_proj.weight.indices"
shard_file = index["weight_map"][tensor_name]

with safe_open(f"models/Model-Trellis-v3/{shard_file}", framework="pt") as f:
    indices = f.get_tensor(tensor_name)
```

## Validation

Verify a v3 model structure:

```bash
python -m metal_marlin.tools.validate_trellis_v3 models/Model-Trellis-v3
```

Checks performed:
- ✅ All shards referenced in index exist
- ✅ All tensors in weight_map present in shards
- ✅ No orphaned tensors in shards
- ✅ Component completeness (all 4 parts per weight)
- ✅ quantization_config.json valid and complete
- ✅ config.json compatible with HuggingFace

## References

- [HuggingFace Safetensors Format](https://huggingface.co/docs/safetensors/index)
- [Trellis Quantization Paper](https://arxiv.org/abs/xxxx.xxxxx)
- Metal Marlin: `metal_marlin/trellis/loader.py`
