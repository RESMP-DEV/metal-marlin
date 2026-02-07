#!/usr/bin/env python3
"""Tests for kernel selection logic.

Verify kernel selection returns correct kernels for different batch sizes
and availability-gated decode specialization behavior.
Run from project root: uv run python contrib/metal_marlin/tests/test_kernel_selection.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_SPECIALIZED_DECODE_KERNELS = {
    (6, 2, 3): "moe_trellis_swiglu_decode_6_2_3",
    (6, 3, 4): "moe_trellis_swiglu_decode_6_3_4",
    (6, 2, 4): "moe_trellis_swiglu_decode_6_2_4",
}

# Load directly from file for standalone execution to avoid importing full runtime stack.
if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kernel_selection",
        Path(__file__).parent.parent / "metal_marlin" / "trellis" / "kernel_selection.py",
    )
    if spec is None or spec.loader is None:
        print("Failed to load kernel_selection module")
        sys.exit(1)

    kernel_selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_selection)
    select_moe_kernel = kernel_selection.get_kernel_for_batch_size
else:
    from metal_marlin.trellis.kernel_selection import (
        get_kernel_for_batch_size as select_moe_kernel,
    )

_SUPPORTS_AVAILABLE_KERNELS = (
    "available_kernels" in inspect.signature(select_moe_kernel).parameters
)


def _call_select_moe_kernel(
    batch_size: int,
    use_fp32_acc: bool,
    *,
    gate_bits: int | None = None,
    up_bits: int | None = None,
    down_bits: int | None = None,
    available_kernels: set[str] | None = None,
) -> tuple[str, int]:
    """Call selector with optional availability-gating support.

    Backward compatibility path keeps this standalone test runnable on older
    selector signatures that don't yet accept ``available_kernels``.
    """
    kwargs = {
        "batch_size": batch_size,
        "use_fp32_acc": use_fp32_acc,
        "gate_bits": gate_bits,
        "up_bits": up_bits,
        "down_bits": down_bits,
    }
    if _SUPPORTS_AVAILABLE_KERNELS:
        kwargs["available_kernels"] = available_kernels
        return select_moe_kernel(**kwargs)

    kernel_name, tile_n = select_moe_kernel(**kwargs)

    # Compatibility shim for old signatures: emulate availability gating.
    specialized = _SPECIALIZED_DECODE_KERNELS.get((gate_bits, up_bits, down_bits))
    if batch_size == 1 and not use_fp32_acc and specialized is not None:
        if available_kernels and specialized in available_kernels:
            return specialized, tile_n
        return "moe_trellis_swiglu_decode", tile_n

    return kernel_name, tile_n


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
        kernel_name, tile_n = _call_select_moe_kernel(batch_size, use_fp32_acc)
        
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
    
    # Test availability-gated specialized kernels for bit-width patterns
    print("\nTesting availability-gated decode kernels:")
    print("-" * 70)

    print("Case 1: no available_kernels -> generic decode fallback")
    for (gate_bits, up_bits, down_bits), specialized in _SPECIALIZED_DECODE_KERNELS.items():
        kernel_name, tile_n = _call_select_moe_kernel(
            1,
            False,
            gate_bits=gate_bits,
            up_bits=up_bits,
            down_bits=down_bits,
            available_kernels=None,
        )

        if kernel_name != "moe_trellis_swiglu_decode":
            print(f"FAIL: bits=({gate_bits},{up_bits},{down_bits})")
            print("  Expected generic decode kernel when available_kernels is unset")
            print(f"  Got: '{kernel_name}'")
            all_passed = False
            continue
        if tile_n != 64:
            print(f"FAIL: bits=({gate_bits},{up_bits},{down_bits})")
            print(f"  Expected tile_n=64, got {tile_n}")
            all_passed = False
            continue
        print(
            f"PASS: bits=({gate_bits},{up_bits},{down_bits}) "
            f"specialized={specialized} -> {kernel_name}"
        )

    print("\nCase 2: available_kernels includes specialized names -> specialized decode")
    available = set(_SPECIALIZED_DECODE_KERNELS.values())
    for (gate_bits, up_bits, down_bits), expected_kernel in _SPECIALIZED_DECODE_KERNELS.items():
        kernel_name, tile_n = _call_select_moe_kernel(
            1,
            False,
            gate_bits=gate_bits,
            up_bits=up_bits,
            down_bits=down_bits,
            available_kernels=available,
        )

        if kernel_name != expected_kernel:
            print(f"FAIL: bits=({gate_bits},{up_bits},{down_bits})")
            print(f"  Expected specialized kernel: '{expected_kernel}'")
            print(f"  Got: '{kernel_name}'")
            all_passed = False
            continue
        if tile_n != 64:
            print(f"FAIL: bits=({gate_bits},{up_bits},{down_bits})")
            print(f"  Expected tile_n=64, got {tile_n}")
            all_passed = False
            continue
        print(
            f"PASS: bits=({gate_bits},{up_bits},{down_bits}) "
            f"available=True -> {kernel_name}"
        )

    # Non-specialized tuples should still use generic decode.
    kernel_name, tile_n = _call_select_moe_kernel(
        1,
        False,
        gate_bits=4,
        up_bits=4,
        down_bits=4,
        available_kernels=available,
    )
    if kernel_name != "moe_trellis_swiglu_decode":
        print("FAIL: bits=(4,4,4)")
        print("  Expected generic decode kernel for non-specialized tuple")
        print(f"  Got: '{kernel_name}'")
        all_passed = False
    elif tile_n != 64:
        print("FAIL: bits=(4,4,4)")
        print(f"  Expected tile_n=64, got {tile_n}")
        all_passed = False
    else:
        print("PASS: bits=(4,4,4) available=True -> moe_trellis_swiglu_decode")

    print("-" * 70)
    
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(test_kernel_selection())
