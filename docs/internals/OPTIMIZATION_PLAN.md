# Draft Speed Optimization Plan

## Target
Achieve >2x speedup on draft model inference for speculative decoding.

## Current Status
- Base implementation: ~13.7ms per iteration
- Optimized implementation: ~10.1ms per iteration (1.35x speedup)
- All 28 tests passing

## Optimizations Implemented

### 1. FusedProjection (mmfp4_mtp_head_optimized.py)
- Uses einsum for batched matrix multiplication
- Single kernel launch for all prediction heads
- Better memory access patterns

### 2. FastSpeculationEngine (mmfp4_mtp_head_optimized.py)
- Pre-allocated buffers for tokens, probs, and logits
- Inference mode for maximum speed
- Buffer reuse across calls

### 3. BatchedSpeculationEngine (mmfp4_mtp_head_optimized.py)
- Parallel processing of multiple sequences
- Better GPU utilization for batch_size > 1

### 4. MMFP4DraftModel Integration (mmfp4_draft.py)
- Automatic dtype conversion to match model
- Fast path for single-batch inference
- Proper buffer management

## Files Modified
1. `contrib/metal_marlin/metal_marlin/layers/mmfp4_mtp_head_optimized.py` - Optimized MTP head
2. `contrib/metal_marlin/metal_marlin/speculative/mmfp4_draft.py` - Draft model with dtype fix
3. `contrib/metal_marlin/metal_marlin/speculative/draft.py` - Reset fix

## Verification
Run tests with:
```bash
uv run pytest tests/test_speculative.py -v
```

Run benchmark with:
```bash
uv run python test_draft_speed.py
```
