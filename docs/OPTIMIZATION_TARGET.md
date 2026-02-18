# Decode GEMV Optimization Target

## Problem: Poor Single-Token Decode Performance

The current `dispatch_gemm_fp4` function uses the `marlin_gemm_fp4` kernel for all batch sizes, including M=1 (single-token decode). This is catastrophically inefficient:

### Root Cause

The `marlin_gemm_fp4` kernel uses 64x64 tiles:
- For M=1 decode: Only 1.5% tile utilization (1 row used, 63 rows wasted)
- **98.4% of compute is wasted on zero-padding**
- Grid computation: `grid_m = (1 + 64 - 1) // 64 = 1` (single wave)
- Thread efficiency: 1/64 = 1.5%

### Impact on End-to-End Performance

| Metric | Current | Target |
|--------|---------|--------|
| Decode throughput | ~0.74 tok/s | 2-5 tok/s (target), 10+ tok/s (stretch) |
| Kernel efficiency | 1.5% | 90%+ |
| MoE layer latency | ~125ms | <50ms |

### Proposed Solution

Add M=1 detection to `dispatch_gemm_fp4` and route to `decode_gemv_fp4_wide`:

```python
def dispatch_gemm_fp4(...):
    # ... existing code ...
    
    # DECODE OPTIMIZATION: Use GEMV kernel for M=1
    if M == 1:
        return _dispatch_decode_gemv_fp4(...)
    
    # ... rest of standard GEMM dispatch ...
```

### The Decode Kernel

`decode_gemv_fp4_wide` (in `src/decode_gemv.metal`):
- **TILE_N = 512**: 8x wider than standard GEMM
- **4 columns per thread**: Better instruction-level parallelism  
- **No M-padding**: 100% utilization for M=1
- **Expected speedup**: ~3-4x for decode

Kernel signature:
```metal
kernel void decode_gemv_fp4_wide(
    device const half* A,      // [1, K]
    device const uint* B,      // [K/8, N] packed FP4
    device const half* scales, // [K/group_size, N]
    device half* C,            // [1, N]
    constant uint& M, constant uint& N, constant uint& K, constant uint& group_size
)
```

### Implementation Plan

1. **Add `_dispatch_decode_gemv_fp4` helper function**:
   - Handles buffer setup for decode kernel
   - Uses TILE_N=512 for grid computation
   - Minimal padding (N only, since M=1)

2. **Modify `dispatch_gemm_fp4`**:
   - Check if M == 1 at start
   - Route to decode helper if true
   - Otherwise use standard GEMM path

3. **Verify**:
   - Run `bench_e2e_decode.py` before/after
   - Target: >2 tok/s minimum, ideally 3-5 tok/s

### Files to Modify

- `metal_marlin/metal_dispatch.py`: Add decode dispatch path

### Verification Command

```bash
cd contrib/metal_marlin
PYTHONPATH=. uv run python benchmarks/bench_e2e_decode.py --max-new-tokens 10
```

Success criteria:
- Before: ~0.74 tok/s
- After: >2.0 tok/s (2.7x improvement)
