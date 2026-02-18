
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, ANY
import torch
from pathlib import Path

# Add the project root to the python path
_parent = Path(__file__).parent
while _parent.name != "metal_marlin" and _parent.parent != _parent:
    _parent = _parent.parent
if _parent.name == "metal_marlin":
    sys.path.insert(0, str(_parent.parent))

from contrib.metal_marlin.metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline, StreamingOutput

class MockModel:
    def __init__(self):
        self.device = "cpu"
        self.config = MagicMock()
        self.config.hidden_size = 128
        self.config.vocab_size = 1000
        
    def to(self, device):
        return self
    
    def eval(self):
        return self
        
    def generate(self, input_ids, **kwargs):
        # Simulate generation by putting tokens into the streamer
        streamer = kwargs.get("streamer")
        if streamer:
            # Simulate generating 5 tokens
            for i in range(5):
                # token_ids must be a tensor
                token = torch.tensor([[i + 10]])
                streamer.put(token)
            streamer.end()
        
        # Return dummy output
        return torch.tensor([[1, 2, 3]])

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "</s>"
    
    def decode(self, token_ids, **kwargs):
        return "".join([f"t{t}" for t in token_ids])
        
    def encode(self, text, **kwargs):
        return [1, 2, 3] # dummy
        
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

class TestStreamingGenerate(unittest.TestCase):
    def test_streaming_generate(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        pipeline = MMFP4Pipeline(model, tokenizer, enable_persistent_cache=False)
        
        # We need to run the async method in an event loop
        async def run_test():
            input_ids = torch.tensor([[1]])
            attention_mask = torch.tensor([[1]])
            
            outputs = []
            async for output in pipeline._streaming_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                top_p=0.9,
            ):
                outputs.append(output)
            
            return outputs

        outputs = asyncio.run(run_test())
        
        # Verify outputs
        self.assertTrue(len(outputs) > 0)
        for output in outputs:
            self.assertIsInstance(output, StreamingOutput)
            print(f"Output: text='{output.text}', reason='{output.finish_reason}', tokens={output.token_count}")
            
        # Verify that we got text back (mock tokenizer returns "t10", "t11" etc)
        full_text = "".join([o.text for o in outputs])
        self.assertTrue("t10" in full_text)
        
        # Verify optimization: check if token counts match
        total_tokens = sum(o.token_count for o in outputs)
        self.assertEqual(total_tokens, 5) # We simulated 5 tokens

if __name__ == "__main__":
    unittest.main()
