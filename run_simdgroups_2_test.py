#!/usr/bin/env python3
"""Test optimization variant: simdgroups_2"""
import json
import re
import time
from pathlib import Path

# Change to metal_marlin directory for proper imports
kernel_path = Path('/Users/kearm/AlphaHENG/contrib/metal_marlin/src/batched_gemm.metal')
original = kernel_path.read_text()

# Apply transformation
pattern = r'constant\s+constexpr\s+uint\s+SIMDGROUPS_PER_TG\s*=\s*\d+'
replacement = r'constant constexpr uint SIMDGROUPS_PER_TG = 2'
modified = re.sub(pattern, replacement, original, count=1)

if modified == original:
    print('ERROR: Pattern did not match')
    result = {'variant': 'simdgroups_2', 'hash': '04e92e0e', 'compile_success': False, 'error': 'Pattern did not match'}
else:
    # Write modified kernel
    kernel_path.write_text(modified)

    try:
        from metal_marlin._compat import HAS_MPS, torch
        from metal_marlin.kernels import pack_fp4_weights
        from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4

        if not HAS_MPS:
            raise RuntimeError('MPS not available')

        # Recompile with modified source
        lib = MetalKernelLibrary.from_source_dir()

        problem_sizes = [(256, 4096, 4096), (32, 4096, 4096), (1, 4096, 4096)]
        results = {}

        for M, N, K in problem_sizes:
            A = torch.randn(M, K, dtype=torch.float16, device='mps')
            weight = torch.randn(N, K, dtype=torch.float16, device='mps')
            B_packed, scales = pack_fp4_weights(weight, group_size=32)

            # Warmup
            for _ in range(5):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(20):
                _ = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, 32)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start

            avg_us = (elapsed / 20) * 1e6
            results[(M, N, K)] = avg_us
            print(f'  {M}x{N}x{K}: {avg_us:.2f} us')

        result = {
            'variant': 'simdgroups_2',
            'hash': '04e92e0e',
            'compile_success': True,
            'results': {str(k): v for k, v in results.items()},
            'timestamp': time.time(),
        }
        print('simdgroups_2: SUCCESS')

    except Exception as e:
        import traceback
        result = {'variant': 'simdgroups_2', 'hash': '04e92e0e', 'compile_success': False, 'error': str(e), 'traceback': traceback.format_exc()}
        print(f'simdgroups_2: FAILED - {e}')
    finally:
        # Restore original
        kernel_path.write_text(original)

# Save result
out_dir = Path('agent_workspace/opt_20260128_205519')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'simdgroups_2.json').write_text(json.dumps(result))
print(f'Saved to {out_dir / "simdgroups_2.json"}')
