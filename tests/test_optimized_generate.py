
from unittest.mock import MagicMock

import torch
from metal_marlin.inference.mmfp4_pipeline import _optimized_generate


def test_optimized_generate():
    torch.manual_seed(42)
    device = "cpu"
    batch_size = 1
    seq_len = 5
    max_new_tokens = 5
    vocab_size = 10
    
    # Mock model
    model = MagicMock()
    # Mock outputs
    # Logits: [batch, 1, vocab]
    # We want deterministic generation.
    # Let's make token 0 have high probability.
    logits = torch.zeros(batch_size, 1, vocab_size, device=device)
    logits[0, 0, 0] = 100.0
    
    outputs = MagicMock()
    outputs.logits = logits
    outputs.past_key_values = None
    outputs.hidden_states = [torch.zeros(batch_size, 1, 10)]
    
    model.return_value = outputs
    
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Run optimized generation with greedy decoding
    out = _optimized_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0
    )
    
    assert out.shape == (batch_size, seq_len + max_new_tokens)
    # Check if generated tokens are 0
    generated = out[:, seq_len:]
    assert (generated == 0).all()
    print("Optimized generate (Greedy) OK")
    
    # Test top-k logic integration
    # Make token 1 have high prob, but token 2 have slightly less.
    # top_k=1 should pick token 1.
    logits = torch.zeros(batch_size, 1, vocab_size, device=device)
    logits[0, 0, 1] = 10.0
    logits[0, 0, 2] = 9.0
    outputs.logits = logits
    
    out = _optimized_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=1,
        temperature=1.0,
        top_k=1,
        top_p=1.0
    )
    assert out[0, -1] == 1
    print("Optimized generate (Top-K) OK")

if __name__ == "__main__":
    try:
        test_optimized_generate()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
