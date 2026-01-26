#!/usr/bin/env python3
"""Inference benchmark for GLM-4.7-Flash on MPS."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "zai-org/GLM-4.7-Flash"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading model (BF16 on MPS)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="mps",
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t0
    print(f"Model load time: {load_time:.2f}s")

    # Warmup
    print("Warming up...")
    messages = [{"role": "user", "content": "Hello"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Benchmark configs
    configs = [
        ("Short prompt", "What is machine learning? Explain briefly.", 64),
        ("Medium prompt", "Write a Python binary search function with docstring.", 256),
        ("Long generation", "Write a tutorial on REST APIs with FastAPI.", 512),
    ]

    print()
    print("=" * 70)
    print("INFERENCE BENCHMARK: zai-org/GLM-4.7-Flash (BF16)")
    print("=" * 70)

    results = []
    for name, prompt, max_tokens in configs:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Run 3 times and average
        times = []
        output_lens = []
        for _ in range(3):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            output_lens.append(output_ids.shape[1] - input_len)

        avg_time = sum(times) / len(times)
        avg_output = sum(output_lens) / len(output_lens)
        tokens_per_sec = avg_output / avg_time

        print(f"{name}:")
        print(f"  Input:  {input_len} tokens")
        print(f"  Output: {avg_output:.0f} tokens")
        print(f"  Time:   {avg_time:.2f}s")
        print(f"  Speed:  {tokens_per_sec:.1f} tok/s")
        print()
        results.append({"name": name, "tps": tokens_per_sec})

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    print(f"Average throughput: {avg_tps:.1f} tok/s")
    print(f"Model load time:    {load_time:.2f}s")


if __name__ == "__main__":
    main()
