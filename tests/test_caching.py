import logging
import os
import sys
from unittest.mock import MagicMock

import torch

# Add contrib/metal_marlin to path
current_dir = os.path.dirname(os.path.abspath(__file__))
contrib_dir = os.path.dirname(current_dir)
sys.path.append(contrib_dir)

from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline



logger = logging.getLogger(__name__)

def test_caching():
    logger.info("running test_caching")
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.config.max_position_embeddings = 16
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pad_token_id = 0
    
    # Mock tokenizer call to return a dict with input_ids
    # pipeline("prompt") -> tokenizer("prompt", return_tensors="pt")
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }
    
    # Mock decode to return "generated text"
    mock_tokenizer.decode.return_value = "generated text"
    
    # Mock generate to return a tensor
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]], dtype=torch.long)

    pipeline = MMFP4Pipeline(
        model=mock_model,
        tokenizer=mock_tokenizer,
        enable_persistent_cache=False,
    )
    
    # Check if cache is initialized (it shouldn't be yet, or maybe it is if I implement it)
    if hasattr(pipeline, "_generation_cache"):
        print("Cache initialized in __init__")
    else:
        print("Cache NOT initialized in __init__")

    print("\n--- First Call ---")
    result1 = pipeline("test prompt")
    print(f"Result 1: {result1}")
    print(f"Generate call count: {mock_model.generate.call_count}")
    
    print("\n--- Second Call (Same prompt) ---")
    result2 = pipeline("test prompt")
    print(f"Result 2: {result2}")
    print(f"Generate call count: {mock_model.generate.call_count}")

    if result1 == result2:
        print("Results match.")
    else:
        print("Results do NOT match.")

    assert result1 == "generated text"
    assert result2 == "generated text"
    assert mock_model.generate.call_count == 1

if __name__ == "__main__":
    test_caching()
