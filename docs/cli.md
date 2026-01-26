# CLI Reference

Complete command-line interface documentation for Metal Marlin.

## Quantize (`hf_loader`)

Download and quantize HuggingFace models.

```bash
python -m metal_marlin.hf_loader MODEL OUTPUT [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `MODEL` | HuggingFace model ID (e.g., `zai-org/GLM-4.7-Flash`) or local path |
| `OUTPUT` | Directory to save quantized model |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--group-size` | 128 | Quantization group size. Smaller = better quality, larger weights |
| `--bits` | 4 | Bits per weight (2-8). Higher bits = better quality for MoE cold experts |
| `--mixed-precision` | auto | Precision preset: `dense`, `moe`, `moe-mtp`, or `auto` |
| `--calibration` | none | Calibration file path for activation-aware quantization |
| `--validate` | true | Compute per-layer quantization error (RMSE) |
| `--token` | none | HuggingFace token for gated models |

**Example output:**

```
======================================================================
METAL MARLIN QUANTIZATION
======================================================================
  Model:      glm4 (47 layers)
  Hidden:     2,048
  MoE:        Yes (64 experts)
  Vocabulary: 154,880
======================================================================

[1/3] Processing embeddings...
[2/3] Processing 47 transformer layers...
[3/3] Processing output layers...

======================================================================
QUANTIZATION COMPLETE
======================================================================
  Original size:   58.42 GB
  Quantized size:  14.61 GB
  Compression:     4.00x
  Mean RMSE:       0.000342
======================================================================
```

## Benchmark (`benchmark_models`)

Compare Metal Marlin against GGUF quantizations.

```bash
python -m metal_marlin.benchmark_models --models MODEL [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | required | Comma-separated HuggingFace model IDs |
| `--output` | `benchmark_results.json` | Output JSON file |
| `--calibration` | `bartowski-v3` | Calibration dataset: `bartowski-v3`, `c4`, or file path |
| `--samples` | 100 | Number of evaluation samples |
| `--max-length` | 512 | Maximum sequence length |
| `--preset` | `auto` | Precision preset: `auto`, `uniform`, `quality`, `speed` |
| `-q, --quiet` | false | Suppress verbose output |

**Example:**

```bash
# Single model with Bartowski calibration
python -m metal_marlin.benchmark_models \
    --models "zai-org/GLM-4.7-Flash" \
    --calibration bartowski-v3

# Multiple models
python -m metal_marlin.benchmark_models \
    --models "zai-org/GLM-4.7-Flash,Qwen/Qwen3-4B" \
    --output comparison.json
```

## Perplexity Evaluation (`eval_perplexity`)

Compute perplexity on a quantized or FP16 model.

```bash
python -m metal_marlin.eval_perplexity MODEL [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `bartowski-v3` | Evaluation dataset |
| `--samples` | 100 | Number of samples |
| `--max-length` | 512 | Maximum sequence length |
| `--batch-size` | 1 | Batch size for evaluation |

## MR-GPTQ Quantization (`mr_gptq`)

Hessian-aware quantization with calibration data.

```bash
python -m metal_marlin.mr_gptq MODEL OUTPUT [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration` | `bartowski-v3` | Calibration dataset |
| `--bits` | 4 | Quantization bits (2-8) |
| `--group-size` | 128 | Group size for per-group quantization |
| `--damp-percent` | 0.01 | Damping for Hessian inverse |
| `--symmetric` | false | Use symmetric quantization |

## Memory Management

Metal Marlin automatically manages RAM during quantization:

1. **Auto-detection**: Queries available system memory
2. **Layer batching**: Estimates per-layer memory requirements
3. **Parallel execution**: Processes multiple layers within RAM budget

**Memory formula:**

```
parallel_layers = (available_ram * 0.8) / per_layer_memory
```

For a 30B MoE model on 64GB M3 Max:
- Available (80%): ~51 GB
- Parallel layers: 16 (capped)
- Speedup: ~4x over sequential

**Override settings:**

```python
from metal_marlin.hf_loader import convert_model_parallel

stats = convert_model_parallel(
    "zai-org/GLM-4.7-Flash",
    "./output",
    max_workers=8,        # Force 8 parallel layers
    ram_budget_gb=32.0,   # Override RAM detection
)
```
