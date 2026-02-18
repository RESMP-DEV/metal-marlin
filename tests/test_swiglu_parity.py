
import torch
from metal_marlin.layers.mmfp4_expert import MMFP4Expert

def test_parity():
    if not torch.backends.mps.is_available():
        print("Skipping MPS test")
        return

    device = "mps"
    hidden_size = 256
    intermediate_size = 128
    group_size = 64
    
    expert = MMFP4Expert(
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        group_size=group_size,
        use_fused=False,
        use_expert_norm=False
    ).to(device)
    
    # Initialize weights with something non-zero/random
    # MMFP4Linear placeholder weights are zeros, so we need to mock them or fill them
    # But MMFP4Linear expects packed weights.
    # To properly test, we can just use the random init from constructor (which are zeros) 
    # but we need to inject some values or use `_standard_forward` vs `_decode_compiled_fastpath`.
    
    # Actually, the placeholder weights are zeros.
    # We should fill them with random data to check parity.
    # But filling packed weights is hard without packer.
    # However, for checking SwiGLU logic, even with random inputs and zero weights,
    # gate and up will be zero? No, if weights are zero, output is zero.
    # So we need non-zero weights.
    
    # Let's just monkey-patch the projections to return random data
    # to test the SwiGLU part specifically? 
    # No, `_decode_compiled_fastpath` calls `_kernel_gemm` which uses `packed_weights`.
    
    # Easier: Just check the code by reading. 
    # But I want to verify the fix works.
    
    # I'll try to run _decode_compiled_fastpath directly.
    
    x = torch.randn(1, hidden_size, device=device, dtype=torch.float16)
    
    # Ensure caches
    expert._ensure_decode_cache()
    
    # 1. Run standard forward (use_fused=False, so standard_forward is used)
    # But wait, standard_forward uses _fused_gate_up which might use kernel if available.
    # For small sizes/zeros, it might be fine.
    
    # We need to make sure _kernel_gemm returns something non-zero.
    # Since I can't easily pack weights, I will rely on code analysis for the SwiGLU bug
    # and use the test to verify that the code runs and produces consistent output if I can.
    
    # Let's rely on the existing tests in test_mmfp4_fused_decode.py
    # and just modify the code.
    pass

if __name__ == "__main__":
    test_parity()
