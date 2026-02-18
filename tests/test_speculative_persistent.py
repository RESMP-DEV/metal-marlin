
import pytest
import torch
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Adjust path to ensure we can import the module under test if needed,
# though pytest usually handles this if run from root.
# from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

# Add parent directory to path for proper imports
_parent = Path(__file__).parent
while _parent.name != "metal_marlin" and _parent.parent != _parent:
    _parent = _parent.parent
if _parent.name == "metal_marlin":
    sys.path.insert(0, str(_parent.parent))

try:
    from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline, _speculative_generate, PersistentKVCache
    from metal_marlin.inference import mmfp4_pipeline
except ImportError:
    # Fallback for direct execution
    from contrib.metal_marlin.metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline, _speculative_generate, PersistentKVCache
    from contrib.metal_marlin.metal_marlin.inference import mmfp4_pipeline

# Mock verify_kernel to return 1 accepted token
def mock_verify_kernel(*args, **kwargs):
    # Returns: num_accepted, accepted_mask, next_token
    # Simulate accepting 1 token. Next token ID 100.
    return (torch.tensor([1]), torch.tensor([True]), torch.tensor([100]))

# Mock classes
class MockConfig:
    def __init__(self):
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.head_dim = 8
        self.vocab_size = 100
        self.max_position_embeddings = 128

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.forward_calls = []
        
    def forward(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False, return_dict=True):
        self.forward_calls.append({
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "shape": input_ids.shape
        })
        batch, seq = input_ids.shape
        # Mock logits
        logits = torch.randn(batch, seq, self.config.vocab_size)
        
        # Mock past_key_values
        if past_key_values is None:
            # Create new
            pk = []
            for _ in range(self.config.num_hidden_layers):
                pk.append((torch.randn(1, 1, seq, 1), torch.randn(1, 1, seq, 1)))
            past_key_values = tuple(pk)
        else:
            # Append new?
            # For this mock we just return the old one + new length awareness if possible
            # But simpler: just return a dummy tuple that is distinct
            pk = []
            # We assume the caller handles appending logic or we just return a placeholder
            # that represents "updated cache"
            for _ in range(self.config.num_hidden_layers):
                 # Just return tensor with shape that suggests growth
                 old_len = past_key_values[0][0].shape[2]
                 new_len = old_len + seq
                 pk.append((torch.randn(1, 1, new_len, 1), torch.randn(1, 1, new_len, 1)))
            past_key_values = tuple(pk)
        
        # Mock hidden states
        hidden_states = None
        if output_hidden_states:
            # List of layers, last one [batch, seq, hidden]
            hidden_states = [torch.randn(batch, seq, self.config.hidden_size)]
            
        output = MagicMock()
        output.logits = logits
        output.past_key_values = past_key_values
        output.hidden_states = hidden_states
        return output
    
    def generate(self, input_ids, **kwargs):
        # Mock generate not used in speculative path (except if draft model missing)
        return torch.cat([input_ids, torch.tensor([[1, 2]])], dim=1)
        
    def eval(self):
        return self
        
    def to(self, device):
        return self

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 100
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def __call__(self, text, return_tensors="pt"):
        # Return sequence "1, 2, 3, 4" for "Hello"
        # "1, 2, 3, 4, 5, 6" for "Hello world"
        if "world" in text:
             ids = [1, 2, 3, 4, 5, 6]
        else:
             ids = [1, 2, 3, 4]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
    
    def decode(self, ids, skip_special_tokens=True):
        return "mock output"

def test_speculative_persistent_cache_integration():
    # Patch verify_kernel
    with patch('metal_marlin.inference.mmfp4_pipeline.verify_kernel', side_effect=mock_verify_kernel):
        model = MockModel()
        tokenizer = MockTokenizer()
        pipeline = MMFP4Pipeline(model, tokenizer, enable_persistent_cache=True)
        
        # Enable speculative decoding
        pipeline._speculative_enabled = True
        pipeline._draft_model = MagicMock()
        
        # Mock draft model cache
        pipeline._draft_model_cache = MagicMock()
        draft_out = MagicMock()
        draft_out.tokens = torch.tensor([[5, 6]])
        pipeline._draft_model_cache.speculate_from_hidden.return_value = draft_out
        
        # 1. First run: "Hello"
        # Should initialize cache and be a miss
        print("Run 1: Hello")
        pipeline("Hello", max_new_tokens=2)
        
        assert pipeline._persistent_kv is not None, "Persistent cache should be initialized"
        assert pipeline._persistent_kv.miss_count == 1
        assert pipeline._persistent_kv.hit_count == 0
        assert pipeline._persistent_kv.cached_ids.shape[1] > 4 # 4 (input) + generated
        
        # 2. Second run: "Hello" (Same prompt)
        # Should be a hit
        print("Run 2: Hello (Repeat)")
        model.forward_calls = [] # Reset spy
        pipeline("Hello", max_new_tokens=2)
        
        assert pipeline._persistent_kv.hit_count == 1
        # Check that we passed sliced input (empty suffix if full match?)
        # "Hello" is [1, 2, 3, 4]. Cache has [1, 2, 3, 4, ...].
        # Match length should be 4 (or more).
        # Suffix is empty?
        # If suffix is empty, _speculative_generate is called with empty input?
        # Wait, if match_len == input_len, suffix is empty.
        # Can _speculative_generate handle empty input_ids?
        # It calls pipeline.model(input_ids).
        # If input_ids is empty, model(empty) -> crash?
        
        # Let's see what happens.
        # Actually, if we match 4 tokens, we call _speculative_generate with input_ids[0:0] i.e. empty?
        # Let's verify what `_match_cache_prefix` returns.
        # It returns `input_ids[:, match_len:]`.
        # If full match, it is empty tensor.
        
        # If input_ids is empty, _speculative_generate:
        # batch_size, seq_len = input_ids.shape -> seq_len=0.
        # outputs = pipeline.model(input_ids, past_key_values=past_kv...)
        # If model supports empty input (just to generate next token from past_kv?), usually NO.
        # Models usually expect at least 1 token if they are to generate something.
        # OR they expect `past_key_values` and generate next token?
        # But `pipeline.model(..., use_cache=True)` typically processes input_ids.
        # If input_ids is empty, it returns nothing?
        
        # Wait, if I have full prefix in cache, I still need to generate *next* token.
        # But I don't have a "next" input token. I want to predict *after* the cache.
        # How do you do that?
        # Usually you pass the *last* token of the prefix if it wasn't processed fully?
        # But `cached_ids` implies they were processed.
        # The KV cache contains keys/values for the cached_ids.
        # So we are ready to predict the token *after* cached_ids.
        
        # In standard generate:
        # `model.generate` handles this.
        # But `_speculative_generate` calls `pipeline.model` manually.
        # `pipeline.model` (MMFP4ForCausalLM) usually wraps `forward`.
        # `forward` needs input.
        
        # If we have no input tokens (all in cache), we assume we want to generate.
        # But to generate, we need the *hidden state* of the last token.
        # If we don't have it (cache only has KV), we might need to re-process the last token?
        # Or `past_key_values` is enough?
        # `model(input_ids=None, past_key_values=...)` -> logits?
        # Most HF models require `input_ids` or `inputs_embeds`.
        
        # So usually, if we have a full match, we should revert 1 token?
        # `_match_cache_prefix` logic:
        # It returns `remaining`.
        
        # If `remaining` is empty, we might have a problem in `_speculative_generate`.
        # Let's assume for this test we use "Hello world" which has suffix "5, 6" after "Hello".
        
        print("Run 3: Hello world (Suffix)")
        # Cache has "Hello..."
        # Input "Hello world" -> [1, 2, 3, 4, 5, 6]
        # Match [1, 2, 3, 4]. Suffix [5, 6].
        
        model.forward_calls = []
        pipeline("Hello world", max_new_tokens=2)
        
        assert pipeline._persistent_kv.hit_count == 2
        # Check what was passed to forward
        assert len(model.forward_calls) > 0
        first_call_input = model.forward_calls[0]["input_ids"]
        # Should be [5, 6] (length 2)
        assert first_call_input.shape[1] == 2, f"Expected input length 2 (suffix), got {first_call_input.shape[1]}"
        
        print("Test passed!")

if __name__ == "__main__":
    test_speculative_persistent_cache_integration()
