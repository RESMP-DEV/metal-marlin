import json
import re
import sys
import time

sys.path.insert(0, '.')

from pathlib import Path

kernel_path = Path('/Users/kearm/AlphaHENG/contrib/metal_marlin/src/sparse.metal')
original = kernel_path.read_text()

# Apply transformation - match separate lines
# First match TILE_K, then TILE_N
pattern_k = r'(constant\s+constexpr\s+uint\s+TILE_K\s*=\s*)\d+'
pattern_n = r'(constant\s+constexpr\s+uint\s+TILE_N\s*=\s*)\d+'

modified = re.sub(pattern_k, r'\g<1>24', original, count=1)
modified = re.sub(pattern_n, r'\g<1>64', modified, count=1)

if modified == original:
    print('ERROR: Pattern did not match')
    result = {'variant': 'rand_003_t24_t64', 'hash': '879ad057', 'compile_success': False, 'error': 'Pattern did not match'}
else:
    # Write modified kernel
    kernel_path.write_text(modified)
    print("Applied TILE_K=24, TILE_N=64")

    try:
        from metal_marlin._compat import HAS_MPS, torch
        from metal_marlin.kernels import pack_fp4_weights
        from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_gemm_fp4

        if not HAS_MPS:
            raise RuntimeError('MPS not available')

        # Recompile with modified source
        lib = MetalKernelLibrary.from_source_dir()

        problem_sizes = [1,9728,2560], [1,2560,9728], [128,768,2048], [256,1536,2048], [256,2048,7168], [2048,9728,2560]
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
            'variant': 'rand_003_t24_t64',
            'hash': '879ad057',
            'compile_success': True,
            'results': {str(k): v for k, v in results.items()},
            'timestamp': time.time(),
        }
        print('rand_003_t24_t64: SUCCESS')

    except Exception as e:
        result = {'variant': 'rand_003_t24_t64', 'hash': '879ad057', 'compile_success': False, 'error': str(e)}
        print(f'rand_003_t24_t64: FAILED - {e}')
    finally:
        # Restore original
        kernel_path.write_text(original)

# Save result
out_dir = Path('agent_workspace/opt_20260128_205433')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'rand_003_t24_t64.json').write_text(json.dumps(result))
print(f'Saved to {out_dir / "rand_003_t24_t64.json"}')
