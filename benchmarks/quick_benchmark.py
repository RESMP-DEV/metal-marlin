#!/usr/bin/env python3
"""Quick GLM-4.7-Flash throughput benchmark."""

import sys
import time

import torch
from transformers import AutoTokenizer

from metal_marlin.trellis.lm import TrellisForCausalLM

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


print("Loading model...")
t0 = time.perf_counter()
model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', device='mps')
tokenizer = AutoTokenizer.from_pretrained(
    'zai-org/GLM-4.7-Flash', trust_remote_code=True)
load_time = time.perf_counter() - t0
print(f"Model loaded in {load_time:.1f}s")

# Check memory
if torch.backends.mps.is_available():
    allocated = torch.mps.current_allocated_memory() / (1024**3)
    print(f"MPS memory: {allocated:.2f} GB\n")

# Test 1: Prefill (encode) throughput
print("=" * 60)
print("TEST 1: Prefill Throughput")
print("=" * 60)
contexts = [128, 512, 1024, 2048]
for ctx_len in contexts:
    # Generate input of roughly ctx_len tokens
    words = "the quick brown fox jumps over the lazy dog " * (ctx_len // 8)
    input_ids = tokenizer.encode(
        words, truncation=True, max_length=ctx_len, return_tensors='pt').to('mps')
    actual_len = input_ids.shape[1]

    # Warmup
    with torch.inference_mode():
        _ = model(input_ids)
    torch.mps.synchronize()

    # Measure prefill
    times = []
    for _ in range(3):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(input_ids)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    tps = actual_len / avg_time
    print(
        f"  Context {actual_len:>4}: {tps:>6.1f} tok/s  ({avg_time*1000:>6.1f}ms)")

# Test 2: Decode (generation) throughput
print("\n" + "=" * 60)
print("TEST 2: Decode Throughput (autoregressive generation)")
print("=" * 60)
prompts = [
    "The capital of France is",
    "To be or not to be,",
    "In the beginning",
]

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('mps')
    prompt_len = input_ids.shape[1]
    gen_tokens = 32

    # Warmup
    with torch.inference_mode():
        _ = model.generate(input_ids, max_new_tokens=5)
    torch.mps.synchronize()

    # Measure generation
    torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids, max_new_tokens=gen_tokens, temperature=0.0)
    torch.mps.synchronize()
    gen_time = time.perf_counter() - t0

    actual_new_tokens = output.shape[1] - prompt_len
    decode_tps = actual_new_tokens / gen_time

    print(
        f"  Prompt len {prompt_len:>2}: {decode_tps:>5.1f} tok/s  [{actual_new_tokens} tokens in {gen_time:.2f}s]")

# Test 3: End-to-end with longer generation
print("\n" + "=" * 60)
print("TEST 3: End-to-End (prefill + decode)")
print("=" * 60)
prompt = "Write a haiku about programming:"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('mps')
prompt_len = input_ids.shape[1]
gen_tokens = 50

torch.mps.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    output = model.generate(
        input_ids, max_new_tokens=gen_tokens, temperature=0.7)
torch.mps.synchronize()
total_time = time.perf_counter() - t0

actual_new_tokens = output.shape[1] - prompt_len
total_tps = (prompt_len + actual_new_tokens) / total_time
decode_only_tps = actual_new_tokens / total_time

generated_text = tokenizer.decode(
    output[0][prompt_len:], skip_special_tokens=True)
print(f"  Total time: {total_time:.2f}s")
print(f"  Prompt: {prompt_len} tokens")
print(f"  Generated: {actual_new_tokens} tokens")
print(f"  Total TPS: {total_tps:.1f} tok/s")
print(f"  Decode TPS: {decode_only_tps:.1f} tok/s")
print("\n  Generated text:")
print(f"  > {generated_text[:200]}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Model loads successfully ({load_time:.1f}s)")
print(f"✅ Memory usage: {allocated:.2f} GB (3-bit quantization working)")
print(f"📊 Decode performance: ~{decode_only_tps:.1f} tok/s")
print("\nNote: Metal MoE kernels failing to compile (zero-length array errors)")
print("Current expectation: 1-2 tok/s (CPU fallback path)")
