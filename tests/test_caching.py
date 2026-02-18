import sys
import os
from unittest.mock import MagicMock

# Add contrib/metal_marlin to path
current_dir = os.path.dirname(os.path.abspath(__file__))
contrib_dir = os.path.dirname(current_dir)
sys.path.append(contrib_dir)

# Mock torch and transformers before importing anything from metal_marlin
mock_torch = MagicMock()
mock_torch.backends.mps.is_available.return_value = False
mock_torch.cuda.is_available.return_value = False
mock_torch.float16 = "float16"

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()
sys.modules["torch.backends.mps"] = MagicMock()

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["transformers.models"] = MagicMock()
sys.modules["transformers.models.glm4_moe"] = MagicMock()
sys.modules["transformers.models.glm4_moe.modeling_glm4_moe"] = MagicMock()

# Now import the pipeline
from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

def test_caching():
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_tokenizer = MagicMock()
    
    # Mock tokenizer call to return a dict with input_ids
    # pipeline("prompt") -> tokenizer("prompt", return_tensors="pt")
    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    
    # Mock decode to return "generated text"
    mock_tokenizer.decode.return_value = "generated text"
    
    # Mock generate to return a tensor
    mock_model.generate.return_value = [MagicMock()]

    pipeline = MMFP4Pipeline(model=mock_model, tokenizer=mock_tokenizer)
    
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

    # Check if cache was used
    # If implemented, call_count should remain same as after first call
    if mock_model.generate.call_count == 1:
        print("SUCCESS: Cache was used.")
    elif mock_model.generate.call_count == 2:
        print("FAILURE: Cache was NOT used.")
    else:
        print(f"Unexpected call count: {mock_model.generate.call_count}")

if __name__ == "__main__":
    test_caching()