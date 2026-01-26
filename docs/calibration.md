# Calibration Guide

Calibration is the process of collecting activation statistics from representative input data to inform quantization decisions. Proper calibration is the difference between 92% and 98% FP16 quality recovery.

## Quick Start

```bash
# Best quality: MR-GPTQ with Bartowski v3 calibration
python -m metal_marlin quantize \
    --input Qwen/Qwen3-32B \
    --output Qwen3-32B-FP4 \
    --method mr-gptq \
    --calibration bartowski-v3

# Fast iteration: RTN with no calibration
python -m metal_marlin quantize \
    --input Qwen/Qwen3-32B \
    --output Qwen3-32B-RTN \
    --method rtn
```

## Why Calibration Matters

Without calibration (RTN quantization):
- Every weight is treated equally
- Outliers dominate scale factors
- 5-10% quality loss typical

With calibration (GPTQ/MR-GPTQ):
- Weights adjusted based on activation importance
- Quantization error distributed optimally
- 2-3% quality loss typical

## Dataset Selection

### Bartowski v3 (Recommended)

The gold standard for general-purpose LLM calibration.

```python
from metal_marlin.calibration import BartowskiCalibration

# Full dataset (recommended for production)
calibration = BartowskiCalibration.v3()

# Limited samples (faster, for testing)
calibration = BartowskiCalibration.v3(max_samples=128)
```

**Characteristics:**
- ~800 carefully curated samples
- Multi-domain: code, chat, math, reasoning, prose
- Balanced token distribution
- Public domain / permissively licensed

### WikiText-2

Single-domain benchmark dataset. Useful for comparison but **not recommended for calibration**.

```bash
python -m metal_marlin quantize \
    --input model/ \
    --output output/ \
    --method mr-gptq \
    --calibration wikitext2
```

**Problems with WikiText-2:**
- 100% Wikipedia text (formal English prose)
- No code, math, or conversational patterns
- Produces biased Hessian estimates
- Models calibrated on WikiText-2 perform worse on real workloads

| Calibration | WikiText-2 PPL | Code Completion | Chat Quality |
|-------------|----------------|-----------------|--------------|
| WikiText-2 | 7.89 (best) | Poor | Poor |
| **Bartowski v3** | 7.92 | **Good** | **Good** |

WikiText-2 optimizes for WikiText-2 perplexity, not real-world performance.

### C4 (Common Crawl)

Web-scraped text. More diverse than WikiText-2 but still English prose.

```bash
python -m metal_marlin quantize \
    --input model/ \
    --output output/ \
    --method mr-gptq \
    --calibration c4
```

**When to use C4:**
- Fallback when Bartowski v3 is unavailable
- General web content models
- Not ideal for code-focused models

### Custom Datasets

For domain-specific models, custom calibration outperforms generic datasets.

```python
from metal_marlin.calibration import BartowskiCalibration, CalibrationDataset

# From plain text file (blank-line separated)
custom = BartowskiCalibration.from_local("my_samples.txt")

# From JSON array
custom = BartowskiCalibration.from_local("samples.json")
# Expected format: ["sample 1", "sample 2", ...]

# From JSONL
custom = BartowskiCalibration.from_local("samples.jsonl")
# Expected format: {"text": "sample 1"}
#                  {"text": "sample 2"}

# Programmatic construction
dataset = CalibrationDataset(
    samples=["Your domain text...", "Another sample..."],
    name="my_domain",
    version="v1",
)
```

## Building Custom Calibration Data

### Principles

1. **Match deployment distribution**: If 60% of real queries are code, 60% of calibration should be code
2. **Diversity > Volume**: 500 diverse samples beats 5000 repetitive samples
3. **Include edge cases**: Long inputs, rare tokens, special formatting
4. **Quality matters**: Clean, representative text (no duplicates, no noise)

### Recommended Sample Counts

| Use Case | Minimum | Recommended | Maximum Useful |
|----------|---------|-------------|----------------|
| Quick testing | 64 | 128 | 256 |
| Development | 128 | 256 | 512 |
| **Production** | **256** | **512** | **1024** |
| Diminishing returns | >512 | - | - |

Beyond ~512 samples, Hessian estimates stabilize and additional samples provide minimal benefit.

### Domain-Specific Examples

**Code-focused model (DeepSeek-Coder):**
```python
samples = [
    # Python examples
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    # Rust examples
    "fn main() {\n    let mut x = 5;\n    println!(\"x = {x}\");\n}",
    # Include docstrings, comments, tests
    "\"\"\"Module for handling API requests.\"\"\"\nimport requests...",
    # Error messages and debugging
    "Traceback (most recent call last):\n  File \"main.py\", line 10...",
]
```

**Math/reasoning model:**
```python
samples = [
    # Step-by-step reasoning
    "Let's solve this step by step:\n1. First, we identify...",
    # LaTeX math
    "The derivative of $f(x) = x^2$ is $f'(x) = 2x$.",
    # Word problems
    "A train leaves station A at 9 AM traveling at 60 mph...",
]
```

**Chat/assistant model:**
```python
samples = [
    # User-assistant format
    "User: How do I make pasta?\nAssistant: Here's a simple recipe...",
    # Multi-turn conversations
    "User: What's the capital of France?\nAssistant: Paris.\nUser: What about Germany?",
    # Various request types
    "User: Summarize this article...\nUser: Write a poem about...",
]
```

## Activation Range Collection

For advanced use, collect activation ranges directly:

```python
from metal_marlin.calibration import (
    BartowskiCalibration,
    compute_activation_ranges,
    ranges_to_scales,
    save_ranges,
    load_ranges,
)

# Load calibration data
calibration = BartowskiCalibration.v3()

# Collect activation ranges from model
ranges = compute_activation_ranges(
    model_path="path/to/model",
    calibration=calibration,
    max_seq_len=2048,
    verbose=True,
)

# Save for later use
save_ranges(ranges, "activation_ranges.json")

# Convert to quantization scales
scales = ranges_to_scales(ranges, quant_type="fp4")
```

### Analyzing Activation Distributions

```python
import json

# Load saved ranges
with open("activation_ranges.json") as f:
    ranges = json.load(f)

# Find layers with largest ranges (potential outliers)
sorted_by_range = sorted(
    ranges.items(),
    key=lambda x: x[1]["max"] - x[1]["min"],
    reverse=True,
)

print("Layers with largest activation ranges:")
for name, r in sorted_by_range[:10]:
    span = r["max"] - r["min"]
    print(f"  {name}: [{r['min']:.2f}, {r['max']:.2f}] span={span:.2f}")
```

## MoE-Specific Calibration

MoE models require per-expert Hessian collection for optimal results.

```python
from metal_marlin.mr_gptq import MoEMRGPTQQuantizer

quantizer = MoEMRGPTQQuantizer(
    bits=4,
    format="fp4",
    expert_hessian_per_expert=True,  # Key setting
    router_precision="bf16",
)

report = quantizer.quantize_model(
    model_path="GLM-4.7-Flash/",
    calibration_data=BartowskiCalibration.v3(),
    output_path="output/",
)
```

### Why Per-Expert Hessians Matter

In MoE models, each expert sees different input distributions based on routing decisions. A shared Hessian would be dominated by frequently-activated experts, producing suboptimal quantization for rare experts.

| Approach | Shared Expert | Rare Expert | Overall Quality |
|----------|---------------|-------------|-----------------|
| Single global Hessian | Good | Poor | Degraded |
| **Per-expert Hessian** | **Good** | **Good** | **Optimal** |

## Calibration Quality Verification

After quantization, verify calibration quality:

```bash
# Check perplexity
python -m metal_marlin eval \
    --model output/ \
    --metric perplexity \
    --dataset wikitext2

# Compare against reference
python -m metal_marlin eval \
    --model output/ \
    --reference original_model/ \
    --metric kl-divergence
```

### Expected Quality Levels

| Method + Calibration | PPL Degradation | Quality Recovery |
|---------------------|-----------------|------------------|
| RTN (no calibration) | +7-10% | ~92% |
| GPTQ + WikiText-2 | +3-5% | ~95% |
| GPTQ + Bartowski v3 | +2-4% | ~96% |
| **MR-GPTQ + Bartowski v3** | **+2-3%** | **~98%** |

## Troubleshooting

### High quantization error on specific layers

**Symptom:** Certain layers show RMSE 10x higher than average.

**Causes:**
1. Activation outliers in those layers
2. Insufficient calibration diversity
3. Layer handles edge-case inputs

**Solutions:**
- Enable Hadamard rotation (`--method mr-gptq`)
- Increase calibration samples
- Use mixed precision to keep problematic layers in FP16

### Poor downstream task performance

**Symptom:** Low perplexity but bad task accuracy.

**Cause:** Calibration data doesn't match task distribution.

**Solution:** Create custom calibration with task-representative samples.

### Out of memory during calibration

**Symptom:** OOM when collecting Hessians for large models.

**Solutions:**
- Reduce `--samples` count
- Use layer-wise calibration (`--layerwise`)
- Increase swap space / unified memory

## Best Practices Summary

1. **Always use Bartowski v3** for general-purpose models
2. **Create custom datasets** for domain-specific models
3. **Use 256-512 samples** for production quantization
4. **Enable Hadamard rotation** (MR-GPTQ) for best quality
5. **Collect per-expert Hessians** for MoE models
6. **Verify with perplexity** but test on real tasks
7. **Keep routers in BF16** for MoE models

## Further Reading

- [MR-GPTQ Algorithm](mr_gptq.md): Detailed GPTQ and Hadamard rotation explanation
- [MoE Architecture](moe_architecture.md): Expert quantization strategies
- [Mixed Precision](mixed_precision.md): Layer-specific precision configuration
