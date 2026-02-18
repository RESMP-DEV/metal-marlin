import pytest
import torch
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any

from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline, PersistentKVCache

# Mock classes
@dataclass
class MockConfig:
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 16
    max_position_embeddings: int = 128
    vocab_size: int = 100

class MockModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.config = MockConfig()
        self.dtype = torch.float16
        
    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        
        # Mock KV cache
        # Create dummy past_key_values for the current input
        # We assume if past_key_values is passed, we append to it (conceptually), 
        # but for this mock we just return a full new one for simplicity or just the new part
        # depending on what generate expects. 
        # But here we are in forward, usually it returns the NEW kv for the input.
        
        new_past_key_values = []
        for _ in range(self.config.num_hidden_layers):
            k = torch.randn(batch_size, self.config.num_attention_heads, seq_len, self.config.head_dim, device=self.device)
            v = torch.randn(batch_size, self.config.num_attention_heads, seq_len, self.config.head_dim, device=self.device)
            new_past_key_values.append((k, v))
            
        outputs = MagicMock()
        outputs.logits = logits
        outputs.past_key_values = tuple(new_past_key_values)
        return outputs
    
    def generate(self, input_ids, past_key_values=None, **kwargs):
        # Simplified generate for testing pipeline logic
        # Just return input + 1 random token
        batch_size, seq_len = input_ids.shape
        
        # Determine total length (prefix + current input)
        # If past_key_values is provided, input_ids is just the new part.
        
        # Simulate generation of 1 token
        new_token = torch.randint(0, self.config.vocab_size, (batch_size, 1), device=self.device)
        
        # Check if we are using cache (partial input)
        if past_key_values is not None:
            # We don't reconstruct full sequence here in mock, but the pipeline expects
            # sequences to contain at least the new generated part.
            # MMFP4Pipeline.__call__ handles reconstruction:
            # full_sequences = torch.cat([prefix, outputs.sequences], dim=1)
            # So generate should return sequences for the input_ids passed + generated.
            sequences = torch.cat([input_ids, new_token], dim=1)
            
            # For KV cache, we need to pretend we have the full KV for the full sequence
            # We can't easily know the full length here without tracking, but let's assume
            # we just return a dummy big enough KV.
            # However, PersistentKVCache updates cached_ids from full_sequences.
            # And kv_cache from outputs.past_key_values.
            pass
        else:
            sequences = torch.cat([input_ids, new_token], dim=1)

        outputs = MagicMock()
        outputs.sequences = sequences
        
        # Create dummy past_key_values for the sequence
        # The length should match the accumulated sequence length ideally.
        # But for this test, as long as it returns *something*, it should work.
        # We make it large enough to cover potential length checks.
        dummy_len = 20 # Arbitrary
        new_past_key_values = []
        for _ in range(self.config.num_hidden_layers):
            k = torch.randn(batch_size, self.config.num_attention_heads, dummy_len, self.config.head_dim, device=self.device)
            v = torch.randn(batch_size, self.config.num_attention_heads, dummy_len, self.config.head_dim, device=self.device)
            new_past_key_values.append((k, v))
            
        outputs.past_key_values = tuple(new_past_key_values)
        return outputs
        
    def eval(self):
        return self

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 100
        self.eos_token_id = 99
        self.pad_token_id = 99
    
    def __call__(self, text, return_tensors="pt"):
        # Simple mock tokenization
        # Map text to length to simulate different inputs
        if "Different" in text:
             tokens = [5, 6, 7]
        elif "Hello world" in text:
             tokens = [1, 2, 3, 4]
        else:
             tokens = [1, 2, 3] # Default "Hello"
             
        return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones(1, len(tokens))}
        
    def decode(self, token_ids, skip_special_tokens=True):
        return "mock output"

def test_persistent_kv_cache_logic():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("Requires GPU")
        
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = MockModel(device=device)
    tokenizer = MockTokenizer()
    
    # Ensure input_ids in tokenizer are on correct device
    original_call = tokenizer.__call__
    def device_aware_call(*args, **kwargs):
        res = original_call(*args, **kwargs)
        res["input_ids"] = res["input_ids"].to(device)
        res["attention_mask"] = res["attention_mask"].to(device)
        return res
    tokenizer.__call__ = device_aware_call
    
    pipeline = MMFP4Pipeline(model, tokenizer, enable_persistent_cache=True)
    
    # 1. First run: should be a miss (empty cache)
    prompt = "Hello"
    pipeline(prompt, max_new_tokens=5)
    
    assert pipeline._persistent_kv is not None
    # First time match fails because cached_ids is empty
    assert pipeline._persistent_kv.miss_count == 1
    assert pipeline._persistent_kv.hit_count == 0
    
    cached_ids = pipeline._persistent_kv.cached_ids
    assert cached_ids.shape[1] > 0
    
    # 2. Second run: same prompt, should be a hit (full match of prefix)
    pipeline(prompt, max_new_tokens=5)
    assert pipeline._persistent_kv.hit_count == 1
    
    # 3. Third run: prompt + suffix, should be a hit (prefix match)
    # "Hello world" starts with [1, 2, 3] which matches "Hello" [1, 2, 3]
    pipeline("Hello world", max_new_tokens=5)
    assert pipeline._persistent_kv.hit_count == 2
         
    # 4. Fourth run: different prompt, should be a miss
    # "Different" is [5, 6, 7]
    pipeline("Different", max_new_tokens=5)
    assert pipeline._persistent_kv.miss_count >= 2 # 1 (initial) + 1 (mismatch)
