import gc
import logging
from pathlib import Path

import pytest
import torch

logger = logging.getLogger(__name__)


def test_generation_not_gibberish() -> None:
    logger.info("running test_generation_not_gibberish")
    from metal_marlin.model_utils import load_prequantized_mmfp4_model

    model_path = Path(__file__).resolve().parents[1] / "models" / "glm47-flash-mmfp4"
    if not (model_path / "config.json").exists():
        pytest.skip(f"Missing local GLM-4.7 Flash model under {model_path}")

    model = None
    try:
        model, tokenizer = load_prequantized_mmfp4_model(str(model_path), device="mps")

        input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to("mps")

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=5, do_sample=False)

        response = tokenizer.decode(outputs[0])
        print(f"Response: {response}")

        # Basic quality check
        assert len(response) > 20, "Response too short"
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.mps.empty_cache()
