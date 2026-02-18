
import pytest
import torch
from unittest.mock import MagicMock, patch
from metal_marlin.inference.mmfp4_pipeline import _optimized_generate, _fused_sampling

def test_optimized_generate_passes_buffers():
    """Verify that _optimized_generate passes pre-allocated buffers to _fused_sampling."""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("Requires GPU")
    
    device = "cuda" if torch.cuda.is_available() else "mps"
    
    # Mock model
    model = MagicMock()
    # Mock forward pass output
    logits = torch.randn(1, 1, 100, device=device) # [batch, seq, vocab]
    outputs = MagicMock()
    outputs.logits = logits
    outputs.past_key_values = None
    model.return_value = outputs
    
    input_ids = torch.tensor([[1, 2, 3]], device=device)
    max_new_tokens = 2
    
    # We want to check if _fused_sampling is called with buffers
    with patch("metal_marlin.inference.mmfp4_pipeline._fused_sampling") as mock_fused:
        # Side effect to actually perform sampling so loop continues correctly
        def side_effect(probs, temperature=1.0, top_p=0.9, top_k=0, out=None, **kwargs):
            if out is not None:
                out.fill_(1) # Return token 1
            return out
        
        mock_fused.side_effect = side_effect
        
        _optimized_generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.9,
            top_k=0, # Try with top_k=0 first
        )
        
        # Check call args
        assert mock_fused.called
        args, kwargs = mock_fused.call_args
        
        # Verify buffers are passed in kwargs
        # For top_k=0, top_p=0.9, we expect sorted_probs_buffer, sorted_indices_buffer, cumsum_buffer
        # to be passed as keyword arguments.
        
        has_buffers = (
            kwargs.get("_sorted_probs_buffer") is not None or
            kwargs.get("_sorted_indices_buffer") is not None or
            kwargs.get("_cumsum_buffer") is not None
        )
        
        assert has_buffers, "_fused_sampling called without buffers!"

def test_optimized_generate_passes_buffers_topk():
    """Verify that _optimized_generate passes pre-allocated buffers when top_k > 0."""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("Requires GPU")
    
    device = "cuda" if torch.cuda.is_available() else "mps"
    
    model = MagicMock()
    logits = torch.randn(1, 1, 100, device=device)
    outputs = MagicMock()
    outputs.logits = logits
    outputs.past_key_values = None
    model.return_value = outputs
    
    input_ids = torch.tensor([[1, 2, 3]], device=device)
    max_new_tokens = 2
    
    with patch("metal_marlin.inference.mmfp4_pipeline._fused_sampling") as mock_fused:
        def side_effect(probs, temperature=1.0, top_p=0.9, top_k=0, out=None, **kwargs):
            if out is not None:
                out.fill_(1)
            return out
        
        mock_fused.side_effect = side_effect
        
        _optimized_generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.9,
            top_k=10, # top_k > 0
        )
        
        assert mock_fused.called
        args, kwargs = mock_fused.call_args
        
        # For top_k=10, top_p=0.9 (and vocab=100), effective_top_k=10 < 100.
        # We expect topk_buffer, topk_indices_buffer, cumsum_buffer.
        
        has_buffers = (
            kwargs.get("_topk_buffer") is not None or
            kwargs.get("_topk_indices_buffer") is not None or
            kwargs.get("_cumsum_buffer") is not None
        )
        
        assert has_buffers, "_fused_sampling called without buffers for top_k!"

def test_fused_sampling_correctness():
    """Verify _fused_sampling works with buffers."""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("Requires GPU")
        
    device = "cuda" if torch.cuda.is_available() else "mps"
    vocab_size = 100
    batch_size = 2
    
    probs = torch.softmax(torch.randn(batch_size, vocab_size, device=device), dim=-1)
    
    # Allocate buffers
    sorted_probs = torch.empty_like(probs)
    sorted_indices = torch.empty_like(probs, dtype=torch.long)
    cumsum = torch.empty_like(probs)
    
    out = torch.empty(batch_size, 1, dtype=torch.long, device=device)
    
    # Call with buffers
    _fused_sampling(
        probs,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        out=out,
        _sorted_probs_buffer=sorted_probs,
        _sorted_indices_buffer=sorted_indices,
        _cumsum_buffer=cumsum
    )
    
    # Check output shape and validity
    assert out.shape == (batch_size, 1)
    assert (out >= 0).all() and (out < vocab_size).all()
    
