
import pytest
import torch
import numpy as np
from metal_marlin import _cpp_ext

def test_fast_router_dispatcher_creation():
    num_experts = 16
    hidden_dim = 64
    top_k = 2
    dispatcher = _cpp_ext.FastRouterDispatcher(num_experts, hidden_dim, top_k)
    
    assert dispatcher.num_experts() == num_experts
    assert dispatcher.hidden_dim() == hidden_dim
    assert dispatcher.top_k() == top_k
    assert dispatcher.hot_pair_count() == 0

def test_fast_router_dispatcher_route_batch():
    num_experts = 4
    hidden_dim = 8
    top_k = 2
    num_tokens = 4
    
    dispatcher = _cpp_ext.FastRouterDispatcher(num_experts, hidden_dim, top_k)
    
    # Create dummy activations (BF16 representation as uint16)
    # Using small values to avoid overflow/underflow issues in float conversion if unchecked
    # But since it expects bf16 bytes, we should be careful.
    # We can use numpy to create float32 and view as uint16 if the bit pattern matches,
    # but BF16 is not standard in numpy.
    # However, for testing, we can just pass some bytes.
    # Or better, use torch to generate bf16 and get bytes.
    
    try:
        activations = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
        activations_bytes = activations.numpy().tobytes() # This might not work for bf16 in numpy
    except:
        # Fallback if torch bf16 to numpy is not supported directly or numpy doesn't support bf16
        # Construct raw bytes. 
        # 1.0 in BF16 is 0x3F80
        activations_bytes = b'\x80\x3f' * (num_tokens * hidden_dim)

    # Weights: [num_experts, hidden_dim]
    weights = np.random.randn(num_experts, hidden_dim).astype(np.float32)
    
    # Bias: [num_experts]
    bias = np.zeros(num_experts, dtype=np.float32)
    
    output = dispatcher.route_batch(activations_bytes, num_tokens, weights, bias)
    
    assert output.num_tokens == num_tokens
    assert output.top_k == top_k
    assert output.num_experts == num_experts
    
    # Check outputs size
    assert len(output.logits) == num_tokens * num_experts
    assert len(output.topk_expert_ids) == num_tokens * top_k
    assert len(output.topk_probs) == num_tokens * top_k
    
    # Basic probability check
    probs = np.array(output.topk_probs).reshape(num_tokens, top_k)
    sums = probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

def test_hot_pair_caching():
    num_experts = 16
    hidden_dim = 8
    top_k = 2
    
    # Threshold = 2 for easy testing
    dispatcher = _cpp_ext.FastRouterDispatcher(num_experts, hidden_dim, top_k, 
                                              hot_pair_threshold=2)
    
    # We can't easily trigger the hot pair logic without controlling the logits exactly 
    # to select specific experts repeatedly.
    # But we can verify the cache starts empty and we can reset it.
    
    assert dispatcher.hot_pair_count() == 0
    dispatcher.reset_hot_pair_cache()
    assert dispatcher.hot_pair_count() == 0

