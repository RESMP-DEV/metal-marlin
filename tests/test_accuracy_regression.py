import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metal_marlin.eval.perplexity import compute_perplexity_wikitext

# Default model
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASELINE_FILE = Path(__file__).parent / "fixtures" / "perplexity_baseline.json"

def test_accuracy_regression(monkeypatch):
    """
    Run perplexity on 100 test samples.
    Compare against baseline (stored).
    Fail if delta > 0.01.
    """
    # Monkeypatch load_wikitext2 to fallback to dummy data if real data fails
    import metal_marlin.eval.perplexity
    original_load = metal_marlin.eval.perplexity.load_wikitext2

    def patched_load_wikitext2(max_samples=None):
        try:
            return original_load(max_samples)
        except Exception as e:
            print(f"Failed to load WikiText-2: {e}. Using dummy data.")
            # Use simple English sentences as fallback
            dummy_text = "The quick brown fox jumps over the lazy dog. " * 10
            return [dummy_text] * (max_samples or 100)

    monkeypatch.setattr(metal_marlin.eval.perplexity, "load_wikitext2", patched_load_wikitext2)

    if not BASELINE_FILE.exists():
        pytest.skip(f"Baseline file not found at {BASELINE_FILE}")

    with open(BASELINE_FILE) as f:
        baselines = json.load(f)

    model_name = os.environ.get("METAL_MARLIN_TEST_MODEL", DEFAULT_MODEL)
    if model_name not in baselines:
         pytest.skip(f"No baseline for model {model_name} in {BASELINE_FILE}")

    baseline_ppl = baselines[model_name]

    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        pytest.skip(f"Failed to load model {model_name}: {e}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Define logits_fn for compute_perplexity_wikitext
    # It expects input_ids as np.ndarray and returns logits as np.ndarray
    def logits_fn(input_ids):
        # input_ids: [1, seq_len]
        input_tensor = torch.tensor(input_ids, device=device)
        with torch.no_grad():
            outputs = model(input_tensor)
            # Ensure we return float32 logits for stability
            return outputs.logits.float().cpu().numpy()

    print("Running perplexity evaluation...")
    # Run perplexity on 100 samples
    try:
        result = compute_perplexity_wikitext(
            logits_fn=logits_fn,
            tokenizer=tokenizer,
            max_samples=100,
            verbose=True
        )
    except Exception as e:
        pytest.fail(f"Perplexity computation failed: {e}")

    current_ppl = result["perplexity"]
    print(f"Model: {model_name}")
    print(f"Baseline PPL: {baseline_ppl}")
    print(f"Current PPL:  {current_ppl}")

    # Fail if delta > 0.01
    delta = abs(current_ppl - baseline_ppl)
    print(f"Delta: {delta}")

    if delta > 0.01:
        pytest.fail(f"Perplexity regression! Delta {delta:.4f} > 0.01 (Current: {current_ppl}, Baseline: {baseline_ppl})")
