import pytest
from unittest.mock import MagicMock, ANY
import torch
from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

def test_continuous_batching_method():
    # Mock model
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.config.hidden_size = 32
    mock_model.config.num_hidden_layers = 1
    mock_model.config.num_attention_heads = 4
    mock_model.config.num_key_value_heads = 4
    
    # Mock forward to return logits and past_key_values
    def forward(input_ids, past_key_values=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 100) # vocab 100
        
        # Create dummy past_key_values (tuple of tuple of tensors)
        # 1 layer, (k, v)
        # k: [batch, heads, seq, dim]
        
        current_seq_len = seq_len
        if past_key_values is not None:
             # past_key_values[0][0] is k
             # k shape: [batch, heads, seq, dim]
             current_seq_len += past_key_values[0][0].shape[2]
        
        k = torch.zeros(batch_size, 4, current_seq_len, 8) # head_dim 8
        v = torch.zeros(batch_size, 4, current_seq_len, 8)
        pkv = ((k, v),)
        
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = pkv
        return mock_output
        
    mock_model.side_effect = forward
    mock_model.eval.return_value = mock_model

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "output"

    pipeline = MMFP4Pipeline(mock_model, mock_tokenizer)
    
    # Requests
    reqs = [
        {"input_ids": torch.tensor([1, 2, 3]), "max_new_tokens": 2},
        {"input_ids": torch.tensor([4, 5]), "max_new_tokens": 1},
    ]
    
    results = list(pipeline._continuous_batching(reqs, batch_size=2))
    
    assert len(results) == 2
    # Check if results match expected format (req_idx, text)
    indices = sorted([r[0] for r in results])
    assert indices == [0, 1]
    
    # Verify model calls
    # We expect:
    # 1. Prefill req 0 (input len 3)
    # 2. Prefill req 1 (input len 2)
    # 3. Decode batch (size 2) -> Req 1 finishes (max_new=1)
    # 4. Decode batch (size 1) -> Req 0 finishes (max_new=2)
    
    assert mock_model.call_count >= 3
