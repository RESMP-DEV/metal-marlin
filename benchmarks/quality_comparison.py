"""Compare quality between original and quantized models."""

from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from tqdm import tqdm


def _extract_logits(outputs: torch.Tensor | object) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits  # type: ignore[no-any-return]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int = 512,
    device: str = "mps",
) -> float:
    """Compute perplexity on text samples."""
    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Computing perplexity"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokens.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = _extract_logits(outputs)[:, :-1, :]  # Shift for next-token prediction
            targets = input_ids[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        raise ValueError("No tokens processed. Check dataset filtering or max_length.")

    return math.exp(total_loss / total_tokens)


def compare_quality(
    original_path: str,
    quantized_path: str,
    *,
    max_length: int = 512,
    sample_count: int = 100,
    min_length: int = 100,
    device: str = "mps",
) -> None:
    """Compare BF16 and quantized model quality."""
    # Load WikiText-2 test set
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > min_length][:sample_count]

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(original_path)

    # Original model perplexity
    from transformers import AutoModelForCausalLM

    bf16_model = AutoModelForCausalLM.from_pretrained(
        original_path, torch_dtype=torch.bfloat16, device_map=device
    )
    bf16_model.eval()
    ppl_bf16 = compute_perplexity(
        bf16_model, tokenizer, texts, max_length=max_length, device=device
    )
    del bf16_model

    # Quantized model perplexity
    from metal_marlin.inference import load_quantized_model

    quant_model = load_quantized_model(quantized_path)
    quant_model.eval()
    ppl_quant = compute_perplexity(
        quant_model, tokenizer, texts, max_length=max_length, device=device
    )

    print(f"BF16 PPL: {ppl_bf16:.2f}")
    print(f"Quantized PPL: {ppl_quant:.2f}")
    print(f"Degradation: {(ppl_quant - ppl_bf16) / ppl_bf16 * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-path", required=True, help="BF16 model path or HF id.")
    parser.add_argument("--quantized-path", required=True, help="Quantized model path.")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--sample-count", type=int, default=100, help="Number of samples.")
    parser.add_argument("--min-length", type=int, default=100, help="Min chars per sample.")
    parser.add_argument("--device", default="mps", help="Torch device (default: mps).")
    args = parser.parse_args()

    compare_quality(
        args.original_path,
        args.quantized_path,
        max_length=args.max_length,
        sample_count=args.sample_count,
        min_length=args.min_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
