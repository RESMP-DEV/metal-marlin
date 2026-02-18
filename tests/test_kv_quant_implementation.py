"""Test KV cache quantization implementation for attention optimization.

This test verifies that the kv_quant feature is properly implemented in:
- MMFP4MLA layer (contrib/metal_marlin/metal_marlin/layers/mmfp4_mla.py)
- Metal shaders (contrib/metal_marlin/metal_marlin/shaders/attention_mla_fused.metal)

Supports quantization modes:
- "none": No quantization (FP16)
- "fp4": 4-bit floating point (E2M1) - 4x memory savings
- "fp8": 8-bit floating point (E4M3) - 2x memory savings
- "int8": 8-bit integer symmetric - 2x memory savings
"""

from __future__ import annotations

import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
requires_mps = pytest.mark.skipif(
    not HAS_TORCH or not torch.backends.mps.is_available(),
    reason="Requires MPS backend"
)


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestKVQuantImplementation:
    """Test KV cache quantization implementation."""

    def test_mmfp4_mla_accepts_kv_quant_param(self):
        """Test that MMFP4MLA accepts kv_quant parameter."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        # Test all supported quantization modes
        for quant_mode in ["none", "fp4", "fp8", "int8"]:
            layer = MMFP4MLA(
                hidden_size=256,
                num_heads=4,
                num_kv_heads=4,
                q_lora_rank=96,
                kv_lora_rank=64,
                qk_nope_head_dim=32,
                qk_rope_head_dim=32,
                v_head_dim=64,
                group_size=64,
                kv_quant=quant_mode,
                kv_quant_group_size=128,
            )
            
            assert layer.kv_quant == quant_mode
            assert layer.kv_quant_group_size == 128
            
            # Check that _kv_quant_enabled is correctly set
            if quant_mode == "none":
                assert not layer._kv_quant_enabled
            else:
                assert layer._kv_quant_enabled

    def test_quantize_kv_cache_fp4(self):
        """Test FP4 quantization of KV cache."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            group_size=64,
            kv_quant="fp4",
            kv_quant_group_size=64,
        )
        
        # Create sample KV cache
        kv_cache = torch.randn(10, 64, dtype=torch.float16)
        
        # Quantize
        quantized, k_scales, v_scales = layer._quantize_kv_cache(kv_cache)
        
        # Verify FP4 quantization produces packed uint32
        assert quantized.dtype == torch.int32
        assert quantized.shape[1] == 64 // 8  # 8 FP4 values packed per uint32
        assert k_scales.shape[0] == 10  # seq_len
        assert v_scales.shape[0] == 10

    def test_quantize_kv_cache_fp8(self):
        """Test FP8 quantization of KV cache."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            group_size=64,
            kv_quant="fp8",
            kv_quant_group_size=64,
        )
        
        # Create sample KV cache
        kv_cache = torch.randn(10, 64, dtype=torch.float16)
        
        # Quantize
        quantized, k_scales, v_scales = layer._quantize_kv_cache(kv_cache)
        
        # Verify FP8 quantization produces uint8
        assert quantized.dtype == torch.uint8
        assert quantized.shape == kv_cache.shape
        assert k_scales.shape[0] == 10
        assert v_scales.shape[0] == 10

    def test_quantize_kv_cache_int8(self):
        """Test INT8 quantization of KV cache."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            group_size=64,
            kv_quant="int8",
            kv_quant_group_size=64,
        )
        
        # Create sample KV cache
        kv_cache = torch.randn(10, 64, dtype=torch.float16)
        
        # Quantize
        quantized, k_scales, v_scales = layer._quantize_kv_cache(kv_cache)
        
        # Verify INT8 quantization produces int8
        assert quantized.dtype == torch.int8
        assert quantized.shape == kv_cache.shape
        assert k_scales.shape[0] == 10
        assert v_scales.shape[0] == 10

    def test_quantize_kv_cache_none(self):
        """Test that 'none' quantization returns original cache with unit scales."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            group_size=64,
            kv_quant="none",
        )
        
        # Create sample KV cache
        kv_cache = torch.randn(10, 64, dtype=torch.float16)
        
        # Quantize (should return as-is with unit scales)
        result, k_scales, v_scales = layer._quantize_kv_cache(kv_cache)
        
        # Verify no quantization - returns original
        assert result is kv_cache
        assert k_scales.shape[0] == 10
        assert torch.allclose(k_scales, torch.ones_like(k_scales))

    def test_mla_attention_params_with_kv_quant(self):
        """Test MLAAttentionParams includes KV quant params."""
        from metal_marlin.mla_fused import MLAAttentionParams
        
        params = MLAAttentionParams(
            batch=1,
            seq_q=1,
            seq_k=8,
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            kv_lora_rank=64,
            q_lora_rank=96,
            rope_dim=32,
            scale=0.125,
            is_causal=True,
            q_a_group_size=64,
            q_b_group_size=64,
            kv_a_group_size=64,
            kv_b_group_size=64,
            o_group_size=64,
            rope_theta=10000.0,
            rope_ratio=1.0,
            rope_base_seq_len=0,
            cache_start_pos=0,
            cache_len=8,
            max_cache_len=32,
            use_fused_q_proj=True,
            use_fused_kv_proj=True,
            fuse_rope_in_kv_a=True,
            skip_kv_decompress=False,
            kv_quant_mode="fp8",
            kv_quant_group_size=128,
            sliding_window=0,
        )
        
        # Verify params are set
        assert params.kv_quant_mode == "fp8"
        assert params.kv_quant_group_size == 128
        
        # Verify struct conversion includes quant mode
        struct_arr = params.to_struct()
        # kv_quant_mode should be mapped: fp8 -> 2
        assert struct_arr[34] == 2  # kv_quant_mode mapped to int
        assert struct_arr[35] == 128  # kv_quant_group_size

    def test_kv_quant_params_in_struct_all_modes(self):
        """Test that all KV quant modes are correctly mapped in struct."""
        from metal_marlin.mla_fused import MLAAttentionParams
        
        mode_map = {
            "none": 0,
            "fp4": 1,
            "fp8": 2,
            "int8": 3,
        }
        
        for mode, expected_int in mode_map.items():
            params = MLAAttentionParams(
                batch=1,
                seq_q=1,
                seq_k=8,
                hidden_size=256,
                num_heads=4,
                head_dim=64,
                kv_lora_rank=64,
                q_lora_rank=96,
                rope_dim=32,
                scale=0.125,
                is_causal=True,
                q_a_group_size=64,
                q_b_group_size=64,
                kv_a_group_size=64,
                kv_b_group_size=64,
                o_group_size=64,
                rope_theta=10000.0,
                rope_ratio=1.0,
                rope_base_seq_len=0,
                cache_start_pos=0,
                cache_len=8,
                max_cache_len=32,
                use_fused_q_proj=True,
                use_fused_kv_proj=True,
                fuse_rope_in_kv_a=True,
                skip_kv_decompress=False,
                kv_quant_mode=mode,
                kv_quant_group_size=64,
                sliding_window=0,
            )
            
            struct_arr = params.to_struct()
            assert struct_arr[34] == expected_int, f"Mode {mode} should map to {expected_int}"

    def test_sliding_window_with_kv_quant(self):
        """Test sliding window attention works with KV quant."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            kv_quant="fp8",
            sliding_window=4096,
        )
        
        assert layer.kv_quant == "fp8"
        assert layer.sliding_window == 4096
        assert layer._kv_quant_enabled


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestKVQuantShaderSupport:
    """Test that Metal shaders support KV quantization."""

    def test_shader_quantization_modes_defined(self):
        """Test that shader quantization mode constants are defined."""
        # Read the shader file to verify quantization support
        import os
        
        shader_path = os.path.join(
            os.path.dirname(__file__),
            "../metal_marlin/shaders/attention_mla_fused.metal"
        )
        
        if os.path.exists(shader_path):
            with open(shader_path) as f:
                shader_content = f.read()
            
            # Verify dequantization functions exist
            assert "dequant_fp4" in shader_content
            assert "dequant_fp8_e4m3" in shader_content
            assert "dequant_int8_sym" in shader_content
            assert "load_kv_quantized" in shader_content
            
            # Verify kv_quant_mode struct field
            assert "kv_quant_mode" in shader_content
            assert "kv_quant_group_size" in shader_content

    def test_mla_write_kv_cache_quantized_kernel_exists(self):
        """Test that the quantized KV cache write kernel exists."""
        import os
        
        shader_path = os.path.join(
            os.path.dirname(__file__),
            "../metal_marlin/shaders/attention_mla_fused.metal"
        )
        
        if os.path.exists(shader_path):
            with open(shader_path) as f:
                shader_content = f.read()
            
            # Verify the quantized cache write kernel exists
            assert "mla_write_kv_cache_quantized" in shader_content


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestKVQuantMemorySavings:
    """Test memory savings from KV cache quantization."""

    def test_fp4_memory_savings(self):
        """Test FP4 provides 4x memory savings."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            kv_quant="fp4",
        )
        
        kv_cache = torch.randn(100, 64, dtype=torch.float16)
        quantized, _, _ = layer._quantize_kv_cache(kv_cache)
        
        # FP4: 4 bits per value = 0.5 bytes per value
        # Original: 2 bytes per value
        # Savings: 4x
        original_bytes = kv_cache.numel() * 2  # FP16 = 2 bytes
        quantized_bytes = quantized.numel() * 4  # int32 = 4 bytes
        
        # Should have ~4x fewer elements (8 values packed per int32)
        assert quantized.numel() == kv_cache.numel() // 8

    def test_fp8_memory_savings(self):
        """Test FP8 provides 2x memory savings."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        layer = MMFP4MLA(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            kv_quant="fp8",
        )
        
        kv_cache = torch.randn(100, 64, dtype=torch.float16)
        quantized, k_scales, _ = layer._quantize_kv_cache(kv_cache)
        
        # FP8: 1 byte per value (vs 2 bytes for FP16) = 2x savings
        # Plus scales overhead (~1-2%)
        original_bytes = kv_cache.numel() * 2
        quantized_bytes = quantized.numel() * 1 + k_scales.numel() * 2
        
        # Should have same number of elements, but 1 byte each
        assert quantized.shape == kv_cache.shape
        assert quantized.dtype == torch.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
