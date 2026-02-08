# Mixed Bit-Width Inference Developer Guide

**Version:** 1.0  
**Last Updated:** 2025-01-21  
**Target Models:** GLM-4.7-Flash, Qwen2-MoE, other Trellis-quantized MoE models

## Overview

Mixed bit-per-weight (mixed BPW) inference allows Mixture of Experts (MoE) models to use different quantization bit-widths for different projections within each expert. This enables:

- **Reduced model size**: Less sensitive layers can use lower precision (2-3 bits)
- **Maintained quality**: Critical layers use higher precision (4-6 bits)
- **Improved throughput**: Specialized kernels for common bit-width combinations
- **Flexible deployment**: Trade off between size, speed, and quality per use case

In GLM-4.7-Flash, mixed BPW achieves:
- **17% decode latency improvement** vs uniform quantization
- **~3.0 average BPW** vs 4.0 for uniform quantization
- **Stable fallback paths** for edge cases

---

## 1. Architecture Overview

### 1.1 How Mixed BPW Works in GLM-4.7-Flash

GLM-4.7-Flash uses a Trellis-quantized MoE architecture with per-expert bit-width allocation:

```
Model: GLM-4.7-Flash-Trellis-MM
├── 32 Layers
│   ├── Attention (uniform 4-bit)
│   │   ├── QKV projection
│   │   └── Output projection
│   └── MoE FFN (mixed BPW)
│       ├── Router (6-bit gate)
│       ├── Shared Experts (6/2/3 bit tuple)
│       └── 64 Routed Experts (mixed tuples)
│           ├── Expert 0-31: (6, 2, 3) bits  [gate, up, down]
│           ├── Expert 32-47: (6, 3, 4) bits
│           └── Expert 48-63: (6, 2, 4) bits
```

**Bit-Width Tuples:** Each expert has 3 projections:
- `gate_bits`: Router input → expert selection weight
- `up_bits`: Input → intermediate activation weight (after gate)
- `down_bits`: Intermediate → output weight (after SwiGLU)

**Dominant Tuples in GLM-4.7-Flash:**
| Tuple | Count | Percentage | Kernel | Use Case |
|-------|-------|------------|--------|----------|
| (6, 2, 3) | 31 | 48.4% | `moe_trellis_swiglu_decode_6_2_3` | Highest sensitivity (router 6-bit, up 2-bit, down 3-bit) |
| (6, 3, 4) | 13 | 20.3% | `moe_trellis_swiglu_decode_6_3_4` | Balanced precision |
| (6, 2, 4) | 2 | 3.1% | `moe_trellis_swiglu_decode_6_2_4` | Conservative down projection |

### 1.2 Bit-Width Allocation Strategy

Bit-width allocation is determined during quantization based on **sensitivity analysis**:

```python
# Pseudo-code for bit-width allocation
def allocate_bits_per_expert(expert_weights, target_avg_bpw=3.0):
    """
    Allocate (gate_bits, up_bits, down_bits) to minimize reconstruction error.
    
    Args:
        expert_weights: Dict of weight tensors for each projection
        target_avg_bpw: Target average bits per weight across all experts
    
    Returns:
        Dict mapping expert_id -> (gate_bits, up_bits, down_bits)
    """
    # Step 1: Compute Hessian (second-order sensitivity) for each projection
    sensitivities = compute_hessians(expert_weights)
    
    # Step 2: Rank experts by sensitivity (router, up, down)
    ranked_projections = rank_by_sensitivity(sensitivities)
    
    # Step 3: Allocate bits greedily
    # - Most sensitive (router): 6 bits (higher for routing stability)
    # - Medium sensitivity (up): 2-3 bits
    # - Lower sensitivity (down): 3-4 bits
    allocations = {}
    for expert_id, (gate_sens, up_sens, down_sens) in ranked_projections:
        gate_bits = 6  # Fixed for routing stability
        up_bits = 2 if up_sens < threshold_med else 3
        down_bits = 3 if down_sens < threshold_low else 4
        
        # Verify average BPW constraint
        if compute_avg_bpw(allocations) > target_avg_bpw + epsilon:
            # Reduce bits for lowest sensitivity experts
            reduce_bits_for_least_sensitive(allocations)
        
        allocations[expert_id] = (gate_bits, up_bits, down_bits)
    
    return allocations
```

**Key Factors in Allocation:**

1. **Router Sensitivity**: Gate projections are kept at 6 bits because:
   - Router errors cascade to all downstream experts
   - Misrouting causes incorrect expert selection → large quality loss
   - Router weights are small (< 1M params), so higher precision adds little cost

2. **Up vs Down Projection**:
   - Up projection: Input → intermediate (often 4x hidden_dim)
   - Down projection: Intermediate → hidden_dim
   - Up is typically more sensitive because errors amplify through SwiGLU
   - Down projection can often tolerate lower precision (2-3 bits)

3. **Expert Diversity**: Different experts specialize on different token patterns:
   - Some experts process factual knowledge (need higher precision)
   - Some experts process syntactic patterns (tolerate lower precision)
   - Sensitivity varies per expert, justifying per-expert allocation

### 1.3 Kernel Dispatch Flow

The kernel dispatch pipeline groups experts by bit-width tuple to minimize kernel launches:

```python
# From: metal_marlin/trellis/moe_dispatch.py

def dispatch_mixed_bpw_moe_fairway(
    lib,
    activations,  # [batch, hidden_dim]
    expert_ids,   # [batch, top_k] - which experts each token uses
    expert_probs, # [batch, top_k] - routing probabilities
    bit_group_buffers,  # Dict: (gate_bits, up_bits, down_bits) -> (expert_list, cached_buffers)
    hidden_dim,
    intermediate_dim,
    num_experts,
    top_k,
    buffer_pool,
    use_fp32_acc=True,
):
    """
    Dispatch MoE by grouping experts with same bit-width tuple.
    
    Reduces dispatch calls from O(num_experts * top_k) to O(unique_tuples).
    """
    batch_size = activations.shape[0]
    output_accum = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device="mps")
    
    # Process each unique bit tuple group
    for bit_tuple, (expert_list, cached_buffers) in bit_group_buffers.items():
        # Step 1: Find all tokens that route to experts in this group
        mask = torch.zeros_like(expert_ids, dtype=torch.bool)
        for expert_id in expert_list:
            mask |= (expert_ids == expert_id)
        
        if not mask.any():
            continue  # Skip if no tokens route to this group
        
        # Step 2: Gather inputs for this group
        batch_indices, slot_indices = mask.nonzero(as_tuple=True)
        group_activations = activations[batch_indices]
        group_expert_ids = expert_ids[batch_indices, slot_indices].unsqueeze(1)
        group_expert_probs = expert_probs[batch_indices, slot_indices].unsqueeze(1)
        
        # Step 3: Prepare grouped fairway inputs (sorted by expert)
        sorted_token_ids, expert_offsets, sorted_probs = prepare_fairway_grouped_inputs(
            expert_ids=group_expert_ids,
            expert_probs=group_expert_probs,
            num_experts=num_experts,
        )
        
        # Step 4: Dispatch with specialized kernel
        group_output = dispatch_moe_trellis_swiglu_grouped_fairway(
            lib=lib,
            activations=group_activations,
            sorted_token_ids=sorted_token_ids,
            expert_offsets=expert_offsets,
            sorted_probs=sorted_probs,
            cached_buffers=cached_buffers,
            bit_tuple=bit_tuple,  # (gate_bits, up_bits, down_bits)
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            buffer_pool=buffer_pool,
        )
        
        # Step 5: Accumulate weighted outputs back to batch positions
        for i, batch_idx in enumerate(batch_indices):
            output_accum[batch_idx] += group_output[i].float()
    
    return output_accum.half()
```

**Dispatch Optimization:**

| Optimization | Description | Gain |
|--------------|-------------|------|
| **Bit-tuple grouping** | Group experts by (gate, up, down) bits | 64 experts → 3-4 groups |
| **Sorted token routing** | Sort tokens by expert ID within each group | Better memory coalescing |
| **Specialized kernels** | Compile-time known dequant parameters | 17% decode improvement |
| **Fairway dispatch** | Batched kernel call per group | Reduced Metal API overhead |

**Fallback Path:**
If grouped fairway dispatch fails (e.g., unsupported bit tuple), the system falls back to:
1. Per-expert sequential dispatch (slower but always works)
2. Logs fallback reason for debugging
3. Maintains counters for monitoring

```python
# Fallback tracking (from: metal_marlin/trellis/moe_dispatch.py)

_MIXED_BPW_GROUPING_BASE_COUNTERS: dict[str, int] = {
    "grouping_calls_total": 0,
    "grouping_gpu_primary_success_total": 0,
    "grouping_cpu_fallback_total": 0,
}

# Get diagnostics at runtime
from metal_marlin.trellis.moe_dispatch import get_mixed_bpw_grouping_fallback_diagnostics

diagnostics = get_mixed_bpw_grouping_fallback_diagnostics()
# Returns:
# {
#     "counters": {"grouping_calls_total": 1234, ...},
#     "last_fallback": {
#         "reason_code": "non_mps_inputs",
#         "detail": "expert_ids=cpu expert_probs=mps",
#         "batch_size": 16,
#         "top_k": 2,
#         "num_experts": 64,
#     }
# }
```

---

## 2. Optimization Guide

### 2.1 Tuning for Your Hardware

#### Hardware-Specific Thresholds

Kernel selection thresholds are hardware-dependent. M4 Max settings:

```python
# From: metal_marlin/trellis/kernel_selection.py

M4_MAX_THRESHOLDS = {
    "decode_max": 1,       # decode kernel for batch_size <= 1
    "prefill4_max": 16,    # prefill4 kernel for 2 <= batch_size <= 16
    "base_max": 32,        # base kernel for 17 <= batch_size <= 32
    "large_batch_min": 33,  # large_batch kernel for batch_size >= 33
}

# Tile sizes
TILE_SIZES = {
    "decode": 64,
    "prefill4": 64,
    "base": 64,
    "large_batch": 128,  # Larger tiles for better memory coalescing
}
```

**For Other Apple Silicon:**

| Device | M3 Max | M3 Pro | M2 Ultra | M2 Max | M1 Max |
|--------|--------|--------|----------|--------|--------|
| Decode Max | 1 | 1 | 1 | 1 | 1 |
| Prefill4 Max | 16 | 12 | 16 | 12 | 8 |
| Base Max | 32 | 24 | 32 | 24 | 16 |
| Large Batch Min | 33 | 25 | 33 | 25 | 17 |
| Large Tile | 128 | 96 | 128 | 96 | 64 |

**Tuning Steps:**

1. **Run kernel selection benchmark:**
```bash
cd contrib/metal_marlin
uv run python benchmarks/bench_m4_kernel_selection.py \
    --model-path models/GLM-4.7-Flash-Trellis-3bpw \
    --output results/kernel_selection_my_device.json
```

2. **Analyze results:**
```python
from metal_marlin.trellis.kernel_selection import get_kernel_info

info = get_kernel_info()
print(f"Device: {info['device']}")
print(f"Thresholds: {info['thresholds']}")
```

3. **Update thresholds in `metal_marlin/trellis/kernel_selection.py`:**
```python
# Example: Custom device tuning
if torch.backends.mps.is_available():
    device_name = torch.mps.device_name()
    if "M1 Max" in device_name:
        M1_MAX_THRESHOLDS = {...}
        M4_MAX_THRESHOLDS = M1_MAX_THRESHOLDS  # Override
```

#### Occupancy Tuning

Threadgroup size affects GPU occupancy and memory usage:

```cpp
// From: src/gemm_trellis_moe.metal

// Compile-time configuration
// Build with: -DMOE_SIMDGROUPS_CONFIG=N
// Options: 4 (128 threads), 8 (256 threads), 16 (512 threads)

#ifndef MOE_SIMDGROUPS_CONFIG
#define MOE_SIMDGROUPS_CONFIG 4  // Default: 4 simdgroups = 128 threads
#endif

constant constexpr uint MOE_SIMDGROUPS = MOE_SIMDGROUPS_CONFIG;
constant constexpr uint MOE_THREADS = MOE_SIMDGROUPS * 32;
```

**Guidelines:**

| Configuration | Threads | Threadgroup Memory | Max Occupancy | Best For |
|----------------|---------|-------------------|---------------|----------|
| `MOE_SIMDGROUPS_CONFIG=4` | 128 | 11,328 bytes | 2 TG/core | Memory-bound kernels (decode) |
| `MOE_SIMDGROUPS_CONFIG=8` | 256 | 12,800 bytes | 2 TG/core | Compute-bound kernels (prefill) |
| `MOE_SIMDGROUPS_CONFIG=16` | 512 | 15,872 bytes | 1 TG/core | Large batch kernels |

**Rebuild with custom occupancy:**
```bash
cd contrib/metal_marlin
cmake -DMOE_SIMDGROUPS_CONFIG=8 -B build_test
cmake --build build_test
```

### 2.2 Benchmarking Methodology

#### End-to-End Throughput Benchmark

```bash
cd contrib/metal_marlin

# Single-token decode (most important for chat)
uv run python benchmarks/benchmark_mixed_bpw_decode.py \
    --model-path models/GLM-4.7-Flash-Trellis-3bpw \
    --num-tokens 100 \
    --warmup 10 \
    --output results/decode_benchmark.json

# Batched prefill (for prompt processing)
uv run python benchmarks/bench_glm47_trellis.py \
    --model-path models/GLM-4.7-Flash-Trellis-3bpw \
    --batch-sizes 1 4 8 16 32 \
    --output results/prefill_benchmark.json
```

#### Output Format

```json
{
  "model": "GLM-4.7-Flash-Trellis-3bpw",
  "device": "M4 Max",
  "decode": {
    "ms_per_token": 8213.33,
    "tokens_per_second": 0.1218,
    "percentile_50_ms": 8100.0,
    "percentile_95_ms": 8900.0,
    "percentile_99_ms": 9200.0
  },
  "mixed_bpw": {
    "average_bpw": 3.02,
    "by_precision": {
      "2-bit": 12345678,
      "3-bit": 23456789,
      "4-bit": 34567890,
      "6-bit": 4567890
    }
  },
  "fallback_counters": {
    "grouping_calls_total": 1234,
    "grouping_cpu_fallback_total": 46,
    "grouping_gpu_primary_success_total": 1188
  },
  "fallback_reasons": {
    "[2, 3, 4, 5, 6]": 31,
    "[2, 3, 4, 6]": 13,
    "[2, 3, 6]": 2
  }
}
```

#### Quality Benchmark

```bash
# Evaluate perplexity on validation set
uv run python benchmarks/eval_glm47_flash.py \
    --model-path models/GLM-4.7-Flash-Trellis-3bpw \
    --dataset wikitext \
    --split validation \
    --output results/quality_report.json
```

#### Comparative Analysis

```python
# Compare uniform vs mixed BPW
import json

uniform = json.load(open("results/uniform_4bit.json"))
mixed = json.load(open("results/mixed_3bit.json"))

print("Decode Latency:")
print(f"  Uniform 4-bit: {uniform['decode']['ms_per_token']:.2f} ms/tok")
print(f"  Mixed 3-bit:   {mixed['decode']['ms_per_token']:.2f} ms/tok")
print(f"  Improvement:   {(1 - mixed/uniform) * 100:.1f}%")

print("\nModel Size:")
print(f"  Uniform 4-bit: {uniform['model_size_gb']:.2f} GB")
print(f"  Mixed 3-bit:   {mixed['model_size_gb']:.2f} GB")
print(f"  Reduction:     {(1 - mixed/uniform) * 100:.1f}%")
```

### 2.3 Performance Profiling

#### Metal Profiler Integration

```bash
# Capture Metal GPU traces
xcrun xctrace record \
    --template "Metal GPU Frame" \
    --output trace.trace \
    --launch \
    uv run python -c "
from metal_marlin.inference import pipeline
model = pipeline.load_model('models/GLM-4.7-Flash-Trellis-3bpw')
output = model.generate('Hello world', max_tokens=50)
"

# Open trace in Instruments
open trace.trace
```

**Key Metrics in Trace:**
- **GPU Time**: Actual kernel execution time
- **Memory Bandwidth**: Read/write throughput
- **Occupancy**: Active threadgroups / max threadgroups
- **Cache Hit Rate**: L1/L2 cache efficiency

#### Python Profiling

```python
import time
from metal_marlin.trellis.moe_dispatch import (
    reset_mixed_bpw_grouping_fallback_counters,
    get_mixed_bpw_grouping_fallback_diagnostics,
)

# Reset counters
reset_mixed_bpw_grouping_fallback_counters()

# Run inference with timing
start = time.perf_counter()
output = model.generate(prompt, max_tokens=100)
elapsed = time.perf_counter() - start

# Get diagnostics
diagnostics = get_mixed_bpw_grouping_fallback_diagnostics()
counters = diagnostics["counters"]

print(f"Total time: {elapsed:.2f}s")
print(f"Tokens: {len(output)}")
print(f"Throughput: {len(output)/elapsed:.2f} tok/s")
print(f"Grouping calls: {counters['grouping_calls_total']}")
print(f"GPU success: {counters['grouping_gpu_primary_success_total']}")
print(f"CPU fallback: {counters['grouping_cpu_fallback_total']}")

if diagnostics.get("last_fallback"):
    fallback = diagnostics["last_fallback"]
    print(f"\nLast fallback:")
    print(f"  Reason: {fallback['reason_code']}")
    print(f"  Detail: {fallback['detail']}")
```

#### Bottleneck Analysis

```bash
# Profile individual kernels
uv run python benchmarks/profile_moe_dispatch.py \
    --model-path models/GLM-4.7-Flash-Trellis-3bpw \
    --num-tokens 100 \
    --output results/moe_profile.json

# Analyze memory bandwidth
uv run python benchmarks/profile_moe_memory.py \
    --model-path models/GLM-4.7-Flash-Trellis-3bpw \
    --output results/bandwidth_profile.json
```

**Identifying Bottlenecks:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| High GPU time, low occupancy | Memory bandwidth bound | Increase tile size, use prefill4 |
| High GPU time, high occupancy | Compute bound | Reduce SIMD groups, optimize kernel |
| High CPU fallback rate | Mixed BPW grouping failing | Check tensor devices, verify bit tuples |
| High decode latency | Wrong kernel selected | Verify batch size thresholds |

---

## 3. Integration Guide

### 3.1 Quantizing New Models with Mixed BPW

#### Step 1: Prepare Model

```bash
# Clone or download model
git clone https://huggingface.co/zai-org/GLM-4.7-Flash

# Or use HuggingFace CLI
huggingface-cli download zai-org/GLM-4.7-Flash \
    --local-dir models/GLM-4.7-Flash-base
```

#### Step 2: Sensitivity Analysis

```python
from metal_marlin.calibration import HessianStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-base",
    device_map="cpu",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "models/GLM-4.7-Flash-base",
    trust_remote_code=True,
)

# Calibrate on representative data
calibration_data = [
    "Your calibration text here.",
    # ... more examples
]

streamer = HessianStreamer(
    model,
    tokenizer,
    calibration_data,
    batch_size=4,
)

# Compute Hessians (sensitivity)
hessians = streamer.compute_hessians(
    layers="all",  # or [0, 1, ..., 31] for specific layers
    projections=["gate_proj", "up_proj", "down_proj"],
)

# Save Hessian for quantization
import json
with open("models/GLM-4.7-Flash-base/hessians.json", "w") as f:
    json.dump(hessians, f)
```

#### Step 3: Allocate Bit Widths

```python
from metal_marlin.quantization.trellis_tile import allocate_mixed_bpw

# Load Hessian
import json
with open("models/GLM-4.7-Flash-base/hessians.json") as f:
    hessians = json.load(f)

# Allocate bits
allocations = allocate_mixed_bpw(
    hessians,
    target_avg_bpw=3.0,  # Target 3 bits per weight
    min_bits=2,
    max_bits=6,
    router_bits=6,  # Fixed at 6 bits for routing stability
)

# Example output:
# {
#     "layer.0.mlp.expert.0": (6, 2, 3),  # (gate, up, down) bits
#     "layer.0.mlp.expert.1": (6, 2, 3),
#     "layer.0.mlp.expert.2": (6, 3, 4),
#     ...
# }

# Verify allocation
avg_bpw = compute_average_bpw(allocations, model)
print(f"Average BPW: {avg_bpw:.2f} (target: 3.0)")
```

#### Step 4: Quantize with Trellis

```python
from metal_marlin.quantization.trellis_tile import TrellisQuantizer
from metal_marlin.quantization.ldlq import quantize_layer_trellis

quantizer = TrellisQuantizer(
    bits_range=(2, 8),
    hadamard_transform=True,
    scale_groups="per_tile",
)

quantized_model = quantize_model_trellis(
    model=model,
    allocations=allocations,
    quantizer=quantizer,
    output_path="models/GLM-4.7-Flash-Trellis-3bpw",
    shard_size="2GB",
)

print("Quantization complete!")
print(f"Saved to: {quantized_model}")
```

#### Alternative: Script-Based Quantization

```bash
cd contrib/metal_marlin

# Use provided quantization script
uv run python scripts/quantize_layerwise_metal.py \
    --model-path models/GLM-4.7-Flash-base \
    --output-path models/GLM-4.7-Flash-Trellis-3bpw \
    --method trellis \
    --target-bits 3.0 \
    --mixed-bpw \
    --hessian-path models/GLM-4.7-Flash-base/hessians.json \
    --shard-size 2GB
```

### 3.2 API Reference

#### Loading Mixed BPW Models

```python
from metal_marlin.trellis.lm import TrellisForCausalLM
from transformers import AutoTokenizer

# Load model (auto-detects mixed BPW)
model = TrellisForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-Trellis-3bpw",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "models/GLM-4.7-Flash-Trellis-3bpw",
    trust_remote_code=True,
)
```

#### Kernel Selection API

```python
from metal_marlin.trellis.kernel_selection import (
    get_kernel_for_batch_size,
    get_kernel_info,
    recommend_kernel,
)

# Get kernel for specific batch size
kernel_name, tile_n = get_kernel_for_batch_size(
    batch_size=1,  # Decode path
    use_fp32_acc=False,
    gate_bits=6,
    up_bits=2,
    down_bits=3,
    available_kernels={
        "moe_trellis_swiglu_decode",
        "moe_trellis_swiglu_decode_6_2_3",
        "moe_trellis_swiglu_prefill4",
        # ...
    },
)
# Returns: ("moe_trellis_swiglu_decode_6_2_3", 64)

# Get detailed recommendation
rec = recommend_kernel(batch_size=8, use_fp32_acc=True)
# Returns:
# {
#     "batch_size": 8,
#     "use_fp32_acc": True,
#     "kernel_name": "moe_trellis_swiglu_prefill4_fp32acc",
#     "tile_n": 64,
#     "category": "small_prefill",
#     "description": "Small batch prefill (4-token SIMD)"
# }
```

#### Mixed BPW Dispatch API

```python
from metal_marlin.trellis.moe_dispatch import (
    dispatch_mixed_bpw_moe_fairway,
    get_mixed_bpw_grouping_fallback_diagnostics,
    reset_mixed_bpw_grouping_fallback_counters,
)

# Dispatch (high-level API - used internally by model)
output = dispatch_mixed_bpw_moe_fairway(
    lib=model.metal_lib,
    activations=activations,  # [batch, hidden_dim]
    expert_ids=expert_ids,    # [batch, top_k]
    expert_probs=expert_probs, # [batch, top_k]
    bit_group_buffers=model.bit_group_buffers,
    hidden_dim=model.config.hidden_size,
    intermediate_dim=model.config.intermediate_size,
    num_experts=model.config.num_experts,
    top_k=model.config.moe_top_k,
    buffer_pool=model.buffer_pool,
    use_fp32_acc=True,
)

# Diagnostics API
reset_mixed_bpw_grouping_fallback_counters()

# ... run inference ...

diagnostics = get_mixed_bpw_grouping_fallback_diagnostics()
counters = diagnostics["counters"]

print(f"Grouping calls: {counters['grouping_calls_total']}")
print(f"GPU success: {counters['grouping_gpu_primary_success_total']}")
print(f"CPU fallback: {counters['grouping_cpu_fallback_total']}")
```

#### Inference Pipeline API

```python
from metal_marlin.inference.pipeline_v2 import TransformersMarlinPipeline

# Create pipeline
pipeline = TransformersMarlinPipeline(model, tokenizer)

# Generate text (single prompt)
response = pipeline.generate(
    "Explain quantum computing",
    max_tokens=256,
    temperature=0.7,
)

# Chat interface (multi-turn)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]
response = pipeline.chat(messages, max_tokens=128)

# Streaming generation
for token in pipeline.chat(messages, max_tokens=128, stream=True):
    print(token, end="", flush=True)
```

### 3.3 Troubleshooting Common Issues

#### Issue 1: High CPU Fallback Rate

**Symptoms:**
- Many `grouping_cpu_fallback_total` counters
- Logs show "Mixed-precision MoE detected ... Fused batched dispatch disabled"
- Slow inference speed

**Diagnosis:**
```python
diagnostics = get_mixed_bpw_grouping_fallback_diagnostics()
fallback = diagnostics.get("last_fallback")

if fallback:
    print(f"Reason code: {fallback['reason_code']}")
    print(f"Detail: {fallback['detail']}")
    print(f"Batch size: {fallback['batch_size']}")
```

**Common Causes & Fixes:**

| Reason Code | Detail | Fix |
|--------------|--------|-----|
| `gpu_grouping_unavailable` | `group_tokens_by_expert_full_gpu` missing | Rebuild Metal extensions: `uv run pip install -e .` |
| `non_mps_inputs` | `expert_ids` or `expert_probs` not on MPS | Move tensors to MPS: `expert_ids = expert_ids.to("mps")` |
| `invalid_shape` | `expert_offsets` size mismatch | Verify `num_experts` matches model config |
| `kernel_not_found` | Specialized kernel not compiled | Add kernel to `available_kernels` set |

**Example Fix:**
```python
# Before (wrong)
expert_ids = torch.randint(0, num_experts, (batch_size, top_k))  # CPU
expert_probs = torch.rand(batch_size, top_k).to("mps")  # MPS

# After (correct)
expert_ids = torch.randint(0, num_experts, (batch_size, top_k)).to("mps")
expert_probs = torch.rand(batch_size, top_k).to("mps")
```

#### Issue 2: Wrong Kernel Selected

**Symptoms:**
- Slower than expected performance
- Logs show using generic `moe_trellis_swiglu` instead of specialized kernel
- Batch size threshold mismatch

**Diagnosis:**
```python
from metal_marlin.trellis.kernel_selection import recommend_kernel

# Check what kernel should be selected
rec = recommend_kernel(batch_size=1, use_fp32_acc=False)
print(f"Expected kernel: {rec['kernel_name']}")

# Check actual kernel from model
actual_kernel = model.last_dispatched_kernel
print(f"Actual kernel: {actual_kernel}")
```

**Common Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| `batch_size` mismatch | Verify input shape: `activations.shape[0]` |
| `use_fp32_acc` wrong | Check model config: `model.config.use_fp32_acc` |
| Bit tuple not in `available_kernels` | Rebuild with specialized kernels: `cmake -DSPECIALIZED_DECODE=ON` |
| Wrong thresholds for device | Update `M4_MAX_THRESHOLDS` in `kernel_selection.py` |

**Example Fix:**
```python
# Override thresholds for custom device
from metal_marlin.trellis import kernel_selection

kernel_selection.M4_MAX_THRESHOLDS = {
    "decode_max": 1,
    "prefill4_max": 12,  # Adjusted for M3 Pro
    "base_max": 24,
    "large_batch_min": 25,
}
```

#### Issue 3: NaN/Inf in Output

**Symptoms:**
- Generated text contains `<unk>` tokens or gibberish
- Loss becomes NaN during training/evaluation
- Metal debugger shows NaN in kernels

**Diagnosis:**
```python
from metal_marlin.trellis.nan_guard import get_nan_statistics

stats = get_nan_statistics()
print(f"NaN detected: {stats.nan_detected}")
print(f"NaN stage: {stats.nan_stage}")
print(f"Layer with NaN: {stats.layer_index}")
```

**Common Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Scale values too small | Re-quantize with clipping: `quantize_with_clipping=True` |
| Hadamard transform overflow | Disable for unstable layers: `hadamard_transform=False` |
| FP16 accumulation overflow | Use FP32 accumulation: `use_fp32_acc=True` |
| Codebook scale mismatch | Verify `TrellisCodebook.scale` matches quantization |

**Example Fix:**
```python
# Re-quantize with FP32 accumulation for problematic layers
from metal_marlin.quantization.trellis_tile import quantize_layer_trellis

quantized_layer = quantize_layer_trellis(
    weight=layer.weight,
    bits=3,
    use_fp32_acc=True,  # Force FP32 accumulation
    hadamard_transform=True,
)
```

#### Issue 4: Memory Overflow on M1/M2

**Symptoms:**
- `Metal: Error: buffer is too small`
- OOM (Out of Memory) errors
- Crashes during large batch inference

**Diagnosis:**
```python
import torch

print(f"MPS allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
print(f"MPS reserved: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GB")
print(f"MPS limit: {torch.mps.recommended_max_memory() / 1024**3:.2f} GB")
```

**Fixes:**

1. **Reduce batch size:**
```python
# Instead of batch_size=32, use smaller batches
for batch in split_into_batches(prompts, batch_size=8):
    output = model.generate(batch, max_tokens=128)
```

2. **Use gradient checkpointing (if applicable):**
```python
model.gradient_checkpointing_enable()
```

3. **Reduce KV cache size:**
```python
model.config.kv_cache_dtype = "fp8"  # Instead of fp16
```

4. **Increase memory limit:**
```python
import torch
torch.mps.set_per_process_memory_fraction(0.9)  # 90% of available memory
```

#### Issue 5: Slow First Inference (Cold Start)

**Symptoms:**
- First inference takes 10-20 seconds
- Subsequent inferences are fast (100-200 ms/token)
- Metal shader compilation delay

**Diagnosis:**
```bash
# Check if shaders are pre-compiled
ls -lh models/GLM-4.7-Flash-Trellis-3bpw/*.metallib
```

**Fixes:**

1. **Pre-compile shaders:**
```bash
cd contrib/metal_marlin

# Compile all shaders at model load time
METAL_FORCE_COMPILE=1 uv run python -c "
from metal_marlin.trellis.lm import TrellisForCausalLM
model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw')
print('Shaders compiled')
"
```

2. **Save compiled shaders:**
```python
from metal_marlin.trellis.lm import TrellisForCausalLM

# Load and compile
model = TrellisForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-Trellis-3bpw",
    compile_shaders=True,  # Force compilation
)

# Save compiled shaders
model.save_pretrained(
    "models/GLM-4.7-Flash-Trellis-3bpw-compiled",
    save_shaders=True,
)
```

3. **Warm up with dummy inference:**
```python
# Warm up kernel caches
_ = model.generate(" ", max_tokens=1)
print("Warmed up, ready for real inference")
```

---

## Appendix A: Performance Reference

### A.1 Measured Performance (M4 Max)

| Model | Avg BPW | Decode (ms/tok) | Decode (tok/s) | Prefill (ms/tok) | Prefill (tok/s) |
|-------|---------|-----------------|----------------|------------------|-----------------|
| GLM-4.7-Flash (uniform 4-bit) | 4.0 | 9910.00 | 0.101 | 0.45 | 2222 |
| GLM-4.7-Flash (mixed 3-bit) | 3.02 | 8213.33 | 0.122 | 0.38 | 2632 |
| **Improvement** | **-24.5%** | **-17.1%** | **+20.7%** | **-15.6%** | **+18.4%** |

### A.2 Memory Footprint

| Model | Original (BF16) | Uniform 4-bit | Mixed 3-bit | Savings |
|-------|----------------|---------------|-------------|---------|
| GLM-4.7-Flash | 35.2 GB | 8.8 GB | 6.6 GB | 25% vs 4-bit |

### A.3 Quality Metrics

| Model | Perplexity (WikiText) | BLEU (MT) | MMLU |
|-------|------------------------|-----------|------|
| BF16 baseline | 10.23 | 32.4 | 52.1 |
| Uniform 4-bit | 10.51 (+2.7%) | 31.8 (-1.9%) | 51.4 (-1.3%) |
| Mixed 3-bit | 10.38 (+1.5%) | 32.1 (-0.9%) | 51.8 (-0.6%) |

---

## Appendix B: File Reference

### B.1 Core Files

| File | Description |
|------|-------------|
| `metal_marlin/trellis/kernel_selection.py` | Kernel selection logic |
| `metal_marlin/trellis/moe_dispatch.py` | Mixed BPW dispatch implementation |
| `src/gemm_trellis_moe.metal` | Metal MoE kernels |
| `metal_marlin/quantization/trellis_tile.py` | Trellis quantization |
| `metal_marlin/trellis/lm.py` | Model loading API |

### B.2 Benchmark Files

| File | Description |
|------|-------------|
| `benchmarks/benchmark_mixed_bpw_decode.py` | Decode throughput benchmark |
| `benchmarks/analyze_bpw_distribution.py` | BPW distribution analysis |
| `benchmarks/bench_m4_kernel_selection.py` | Kernel selection benchmark |
| `benchmarks/profile_moe_dispatch.py` | MoE dispatch profiling |

### B.3 Documentation Files

| File | Description |
|------|-------------|
| `docs/reports/glm47_mixed_bpw_optimization.md` | Optimization results |
| `docs/reports/glm47_mixed_bit_fairway_dispatch.md` | Dispatch performance |
| `docs/formats/trellis_v3.md` | Trellis format spec |

---

## Appendix C: Glossary

- **BPW**: Bits Per Weight - average number of bits used to store each model weight
- **MoE**: Mixture of Experts - architecture with multiple expert networks selected by a router
- **Trellis**: Trellis-coded quantization - EXL3-style quantization with codebooks
- **Top-K**: Number of experts selected per token (typically 2-4)
- **Fairway Dispatch**: Batched kernel dispatch that processes multiple experts in one call
- **Hessian**: Second derivative matrix - measures parameter sensitivity
- **Gate/Up/Down**: Three projections in SwiGLU MLP architecture
- **FP32 Acc**: FP32 accumulation - using 32-bit accumulators for numerical stability
- **Decode**: Single-token generation (batch_size=1)
- **Prefill**: Multi-token prompt processing (batch_size>1)
- **Threadgroup**: Metal's name for CUDA thread blocks
- **SIMD Group**: Subset of threads in a threadgroup (32 threads on Apple Silicon)

---

## Support & Contributing

For issues, questions, or contributions:

1. **Bug Reports**: Open an issue with:
   - Model name and version
   - Device info (M1/M2/M3/M4, macOS version)
   - Minimal reproducible code
   - Logs and diagnostics output

2. **Performance Issues**: Include:
   - Benchmark results
   - Metal profiler traces
   - Fallback counter diagnostics

3. **Contributions**: See `CONTRIBUTING.md` for guidelines

---

**End of Developer Guide**
