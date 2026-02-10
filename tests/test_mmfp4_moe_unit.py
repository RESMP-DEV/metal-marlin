
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Mock dependencies
mock_dispatch = MagicMock()
sys.modules["metal_marlin.moe_dispatch"] = mock_dispatch

from metal_marlin.layers.mmfp4_moe import MMFP4Expert, MMFP4MoE


class TestMMFP4MoEUnit:
    @pytest.fixture
    def mock_dispatch_module(self):
        with patch("metal_marlin.layers.mmfp4_moe._get_dispatch_module") as mock:
            dispatch = MagicMock()
            mock.return_value = dispatch
            yield dispatch

    @pytest.fixture
    def moe_layer(self):
        # Create a small MoE layer
        return MMFP4MoE(
            n_experts=4,
            n_experts_per_tok=2,
            hidden_size=32,
            moe_intermediate_size=16,
            group_size=32,
            has_shared_expert=True
        )

    def test_forward_expert_inputs_dtype(self, moe_layer, mock_dispatch_module):
        # Setup inputs
        batch_size = 2
        seq_len = 4
        hidden_size = 32
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32) # Input as float32 to test conversion
        
        # Setup mock dispatch behaviors
        # group_tokens_by_expert_full returns dispatch_info
        dispatch_info = MagicMock()
        # expert_offsets needs to be valid for active_indices logic
        # 4 experts. Offsets: [0, 2, 4, 6, 8] -> 2 tokens per expert
        expert_offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)
        dispatch_info.expert_offsets = expert_offsets
        mock_dispatch_module.group_tokens_by_expert_full.return_value = dispatch_info
        
        # gather_for_experts returns expert_inputs
        # Create inputs as float16 (since hidden_f16 is passed to gather)
        # The gather function usually returns float16 if input is float16
        # But wait, logic in forward:
        # hidden_f16 = hidden_flat.to(torch.float16)
        # expert_inputs = dispatch.gather_for_experts(hidden_f16, dispatch_info)
        # So expert_inputs should be float16 coming out of gather.
        
        # However, let's verify that we are indeed using the tensor that WAS converted
        # (or re-converted if we had the bug where we converted inside loop).
        
        # Actually, the optimization was:
        # Before:
        #   for ...
        #      chunk = expert_inputs[start:end]
        #      expert(chunk.to(float16))  <-- repeated conversion if expert_inputs wasn't float16?
        #      # Wait, gathering from hidden_f16 means expert_inputs IS float16.
        #      # The comment says: "expert_inputs is already float16 as it's gathered from hidden_f16"
        #      # BUT the removed code was: chunk = chunk.to(torch.float16)
        #      # So previous code WAS doing redundant conversion on already float16 data?
        #      # OR expert_inputs was somehow not float16?
        
        # Let's look at the file content I read earlier.
        # L235: expert_inputs = dispatch.gather_for_experts(hidden_f16, dispatch_info)
        # hidden_f16 is float16.
        # So expert_inputs is float16.
        # The removed code:
        #   chunk = expert_inputs[start:end]
        #   chunk = chunk.to(torch.float16)
        
        # Yes, it was a redundant conversion if it was already float16,
        # or maybe it was intended to ensure it.
        # The new code:
        #   expert_inputs_f16 = expert_inputs.to(torch.float16)
        #   for ...
        #     chunk_out = expert(expert_inputs_f16[start:end])
        
        # If expert_inputs is ALREADY float16, .to(float16) is a no-op (check).
        # But if it's a no-op, then moving it outside doesn't save much unless __getitem__ does something?
        # Or maybe gather_for_experts returns something else in some cases?
        
        # Regardless, the task was to move it.
        # To verify, I should check that the expert receives a float16 tensor.
        
        # Mock gather to return a tensor.
        expert_inputs = torch.randn(8, hidden_size, dtype=torch.float16)
        mock_dispatch_module.gather_for_experts.return_value = expert_inputs
        
        # Mock scatter
        mock_dispatch_module.scatter_expert_outputs.return_value = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float16)

        # Mock experts
        for expert in moe_layer.experts:
            expert.forward = MagicMock(return_value=torch.randn(2, hidden_size, dtype=torch.float16)) # 2 tokens
        
        # Run forward
        output = moe_layer(x)
        
        # Check that experts were called
        # And check the input passed to them
        assert moe_layer.experts[0].forward.called
        
        # Get arguments passed to expert 0
        args, _ = moe_layer.experts[0].forward.call_args
        input_tensor = args[0]
        
        assert input_tensor.dtype == torch.float16
        
    def test_active_indices_optimization(self, moe_layer, mock_dispatch_module):
        # Test that we only iterate active experts
        batch_size = 1
        seq_len = 1
        hidden_size = 32
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        dispatch_info = MagicMock()
        # 4 experts.
        # expert 0: 0->0 (count 0)
        # expert 1: 0->2 (count 2) - ACTIVE
        # expert 2: 2->2 (count 0)
        # expert 3: 2->4 (count 2) - ACTIVE
        expert_offsets = torch.tensor([0, 0, 2, 2, 4], dtype=torch.int32)
        dispatch_info.expert_offsets = expert_offsets
        mock_dispatch_module.group_tokens_by_expert_full.return_value = dispatch_info
        
        mock_dispatch_module.gather_for_experts.return_value = torch.randn(4, hidden_size, dtype=torch.float16)
        mock_dispatch_module.scatter_expert_outputs.return_value = torch.randn(1, hidden_size, dtype=torch.float16)

        # Mock experts
        for expert in moe_layer.experts:
            expert.forward = MagicMock(return_value=torch.randn(2, hidden_size, dtype=torch.float16))

        moe_layer(x)
        
        # Expert 0 should NOT be called
        assert not moe_layer.experts[0].forward.called
        # Expert 1 SHOULD be called
        assert moe_layer.experts[1].forward.called
        # Expert 2 should NOT be called
        assert not moe_layer.experts[2].forward.called
        # Expert 3 SHOULD be called
        assert moe_layer.experts[3].forward.called

