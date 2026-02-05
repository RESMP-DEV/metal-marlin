# Optimization Workflow for Metal Marlin

## 1. Identify Blockers

Run the profiler:
cd contrib/metal_marlin
uv run python scripts/profile_quantization.py --layers 5 --shape 4096x4096

## 2. Current Optimization Targets

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Eigendecomposition | 15% | <5% | Cython _eigh_fast |
| LDL decomposition | 10% | <3% | Cython _ldl_fast |
| Hessian (Metal) | 10% | <10% | Done |
| Viterbi (Metal) | 50% | <50% | Done |

## 3. Testing Protocol

After any optimization:
1. uv run python scripts/profile_quantization.py --layers 5
2. uv run pytest tests/ -v -k quantization
3. Compare MSE: should be stable (±5%)

## 4. Commands

Build extensions:
cd contrib/metal_marlin
uv run python setup.py build_ext --inplace

Check extension loaded:
uv run python -c "from metal_marlin.quantization._ldl_fast import block_ldl_fast; print('OK')"

## 5. Qwen3-30B MoE Quantization

### Prerequisites

1. Download model:
   ```bash
   huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B
   ```

2. Build extensions:
   ```bash
   cd contrib/metal_marlin
   uv run python setup.py build_ext --inplace
   ```

### Run Quantization

```bash
uv run python scripts/quantize_qwen3_30b.py \
    --model-path ./models/Qwen3-30B-A3B \
    --output-dir ./models/Qwen3-30B-A3B-Trellis-3bpw \
    --bits 3 \
    --expert-high-bits 4 \
    --expert-low-bits 3 \
    --top-k-experts 8 \
    --calibration-samples 512
```

### Sensitivity-Aware Expert Quantization

MoE models have 64+ experts, but only top-K are activated per token.
We identify the most sensitive experts (highest activation variance)
and quantize them at higher precision:

| Expert Type | Bit Precision | Purpose |
|-------------|---------------|---------|
| Top-K sensitive | 4 bits | High-quality routing |
| Other experts | 3 bits | Memory efficiency |

This achieves better quality/size tradeoff than uniform quantization.
