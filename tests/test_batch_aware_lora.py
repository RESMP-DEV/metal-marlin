import torch
import torch.nn as nn
import pytest
from metal_marlin.layers import MarlinLinear, MixedPrecisionLinear

class TestBatchAwareLoRA:
    def test_marlin_linear_lora_dispatch(self):
        in_features = 32
        out_features = 16
        group_size = 16
        
        # Create base linear layer
        linear = nn.Linear(in_features, out_features, bias=False)
        # Ensure weights are substantial enough to survive quantization
        linear.weight.data.normal_(0, 1.0)
        
        marlin = MarlinLinear.from_linear(linear, group_size=group_size)
        
        # Create input
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        
        # Create LoRA adapters
        rank = 4
        num_adapters = 2
        lora_u = torch.randn(num_adapters, in_features, rank)
        lora_v = torch.randn(num_adapters, rank, out_features)
        
        # Assign adapters: [0, 1, -1, 0]
        indices = torch.tensor([0, 1, -1, 0], dtype=torch.long)
        
        # Run dispatch
        out_marlin = marlin.batch_aware_dispatch(x, lora_u, lora_v, indices)
        
        # Manual verification
        # Base output (using marlin forward to account for quantization error)
        base_out = marlin(x)
        expected = base_out.clone()
        
        # Add LoRA
        # Sample 0: Adapter 0
        expected[0] += (x[0] @ lora_u[0]) @ lora_v[0]
        # Sample 1: Adapter 1
        expected[1] += (x[1] @ lora_u[1]) @ lora_v[1]
        # Sample 2: No adapter
        # Sample 3: Adapter 0
        expected[3] += (x[3] @ lora_u[0]) @ lora_v[0]
        
        assert torch.allclose(out_marlin, expected, atol=1e-4)

    def test_mixed_precision_linear_lora_dispatch(self):
        in_features = 128
        out_features = 16
        
        # Create layer
        mp_linear = MixedPrecisionLinear(in_features, out_features, bias=False)
        
        # Input
        batch_size = 4
        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
        
        # LoRA
        rank = 4
        num_adapters = 2
        lora_u = torch.randn(num_adapters, in_features, rank, dtype=torch.bfloat16)
        lora_v = torch.randn(num_adapters, rank, out_features, dtype=torch.bfloat16)
        
        indices = torch.tensor([0, -1, 1, 0], dtype=torch.long)
        
        # Run
        out = mp_linear.batch_aware_dispatch(x, lora_u, lora_v, indices)
        
        # Verify
        base_out = mp_linear(x)
        expected = base_out.clone()
        
        # 0 -> Adapter 0
        expected[0] += (x[0] @ lora_u[0]) @ lora_v[0]
        # 1 -> None
        # 2 -> Adapter 1
        expected[2] += (x[2] @ lora_u[1]) @ lora_v[1]
        # 3 -> Adapter 0
        expected[3] += (x[3] @ lora_u[0]) @ lora_v[0]
        
        assert torch.allclose(out, expected, atol=1e-3)

    def test_packed_input_dispatch(self):
        """Test MixedPrecisionLinear with packed FP4 input (uint32)."""
        in_features = 128
        out_features = 16
        
        mp_linear = MixedPrecisionLinear(in_features, out_features, bias=False)
        
        # Mock packed input
        batch_size = 4
        packed_in = in_features // 8
        x_packed = torch.randint(0, 2**32, (batch_size, packed_in), dtype=torch.uint32)
        
        # Create input scales (required for FP4)
        n_groups = in_features // 128  # default group_size
        x_scales = torch.randn(batch_size, n_groups, dtype=torch.float16)

        # LoRA (needs to match dequantized dtype, which is FP16 from _dequantize_fp4_input)
        rank = 4
        num_adapters = 2
        lora_u = torch.randn(num_adapters, in_features, rank, dtype=torch.float16)
        lora_v = torch.randn(num_adapters, rank, out_features, dtype=torch.float16)
        
        indices = torch.tensor([0, 1, -1, 0], dtype=torch.long)
        
        # Run
        out = mp_linear.batch_aware_dispatch(x_packed, lora_u, lora_v, indices, input_scales=x_scales)
        
        # Verify
        # Manually dequantize input to check
        x_dequant = mp_linear._dequantize_fp4_input(x_packed, x_scales)
        
        # Base forward (calls _forward_fp4 which dequantizes input and weight)
        base_out = mp_linear(x_packed, input_scales=x_scales)
        expected = base_out.clone()
        
        # 0 -> Adapter 0
        expected[0] += (x_dequant[0] @ lora_u[0]) @ lora_v[0]
        # 1 -> Adapter 1
        expected[1] += (x_dequant[1] @ lora_u[1]) @ lora_v[1]
        # 2 -> None
        # 3 -> Adapter 0
        expected[3] += (x_dequant[3] @ lora_u[0]) @ lora_v[0]
        
        assert torch.allclose(out, expected, atol=1e-3)
