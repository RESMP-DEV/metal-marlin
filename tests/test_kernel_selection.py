#!/usr/bin/env python3
"""Tests for kernel selection logic.

Verify select_moe_kernel returns correct kernels for different batch sizes.
Run from project root: uv run python contrib/metal_marlin/tests/test_kernel_selection.py
"""

import sys
from pathlib import Path

# Only mock when running this file directly, not when imported by pytest
if __name__ == "__main__":
    # Add contrib/metal_marlin to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import only the function we need to avoid full torch import chain
    # This imports from the file directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "moe_dispatch", 
        Path(__file__).parent.parent / "metal_marlin" / "trellis" / "moe_dispatch.py"
    )
    if spec is None or spec.loader is None:
        print("Failed to load moe_dispatch module")
        sys.exit(1)
        
    # We need to mock torch and other deps to test just the selection logic
    class MockTensor:
        pass

    class MockTorch:
        Tensor = MockTensor
        float16 = None
        float32 = None
        int32 = None
        long = None
        device = lambda x, device: None
        @staticmethod
        def zeros(*args, **kwargs):
            return None

    # Mock the dependencies
    sys.modules['torch'] = MockTorch()
    sys.modules['numpy'] = type(sys)('numpy')
    sys.modules['numpy'].uint32 = None
    sys.modules['numpy'].array = lambda x, **kwargs: x
    sys.modules['Metal'] = type(sys)('Metal')

    moe_dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moe_dispatch)

    select_moe_kernel = moe_dispatch.select_moe_kernel
else:
    # When imported by pytest, use normal imports
    from metal_marlin.trellis.moe_dispatch import select_moe_kernel


def test_kernel_selection():
    """Test kernel selection for various batch sizes."""
    
    test_cases = [
        # (batch_size, use_fp32_acc, expected_kernel_substring, expected_tile_n)
        (1, False, "decode", 64),
        (1, True, "decode", 64),  # No fp32acc variant for decode
        (2, False, "prefill4", 64),
        (2, True, "prefill4_fp32acc", 64),
        (8, False, "prefill4", 64),
        (16, False, "prefill4", 64),
        (16, True, "prefill4_fp32acc", 64),
        (17, False, "moe_trellis_swiglu", 64),  # base kernel (not prefill4)
        (17, True, "fp32acc", 64),
        (32, False, "moe_trellis_swiglu", 64),  # base kernel
        (33, False, "large_batch", 128),
        (33, True, "fp32acc", 64),  # fallback to base when fp32acc needed
        (64, False, "large_batch", 128),
        (128, False, "large_batch", 128),
    ]
    
    print("Testing select_moe_kernel batch size selection:")
    print("-" * 70)
    
    all_passed = True
    for batch_size, use_fp32_acc, expected_substr, expected_tile_n in test_cases:
        kernel_name, tile_n = select_moe_kernel(batch_size, use_fp32_acc)
        
        # Check kernel name contains expected substring
        if expected_substr not in kernel_name:
            print(f"FAIL: batch={batch_size}, fp32={use_fp32_acc}")
            print(f"  Expected: '{expected_substr}' in kernel name")
            print(f"  Got: '{kernel_name}'")
            all_passed = False
            continue
        
        # Check tile size
        if tile_n != expected_tile_n:
            print(f"FAIL: batch={batch_size}, fp32={use_fp32_acc}")
            print(f"  Expected tile_n={expected_tile_n}, got {tile_n}")
            all_passed = False
            continue
        
        print(f"PASS: batch={batch_size:3d}, fp32={use_fp32_acc!s:5} -> "
              f"{kernel_name:<40} tile={tile_n}")
    
    print("-" * 70)
    
    # Test specialized kernels for bit-width patterns
    print("\nTesting specialized decode kernels:")
    print("-" * 70)
    
    special_cases = [
        # (gate_bits, up_bits, down_bits, expected_kernel_suffix)
        (6, 2, 3, "decode_6_2_3"),
        (6, 3, 4, "decode_6_3_4"),
        (6, 2, 4, "decode_6_2_4"),
        (4, 4, 4, "decode"),  # No specialization
    ]
    
    for gate_bits, up_bits, down_bits, expected_suffix in special_cases:
        kernel_name, tile_n = select_moe_kernel(
            1, False, gate_bits=gate_bits, up_bits=up_bits, down_bits=down_bits
        )
        
        if not kernel_name.endswith(expected_suffix):
            print(f"FAIL: bits=({gate_bits},{up_bits},{down_bits})")
            print(f"  Expected suffix: '{expected_suffix}'")
            print(f"  Got: '{kernel_name}'")
            all_passed = False
            continue
        
        print(f"PASS: bits=({gate_bits},{up_bits},{down_bits}) -> {kernel_name}")
    
    print("-" * 70)
    
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(test_kernel_selection())
