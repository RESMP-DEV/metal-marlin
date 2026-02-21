"""E2E generation test with memory cleanup."""
import torch
import gc
import pytest


def test_generation_not_gibberish():
    from metal_marlin.model_utils import load_prequantized_mmfp4_model
    
    try:
        model, tokenizer = load_prequantized_mmfp4_model('models/glm47-flash-mmfp4', device='mps')
        
        input_ids = tokenizer.encode("The capital of France is", return_tensors='pt').to('mps')
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        
        response = tokenizer.decode(outputs[0])
        print(f"Response: {response}")
        
        # Basic quality check
        assert len(response) > 20, "Response too short"
    finally:
        # ALWAYS cleanup
        del model
        gc.collect()
        torch.mps.empty_cache()
