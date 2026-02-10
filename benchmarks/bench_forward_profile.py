#!/usr/bin/env python3
"""Profile model forward to identify dispatch overhead."""
import torch
from metal_marlin.trellis.model import TrellisForCausalLM
from transformers import AutoTokenizer

print("Loading model...")
model = TrellisForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-Marlin-MMFP4", device="mps"
)

# Warmup tokenizer + inputs
text = "Hello"
tokenizer = AutoTokenizer.from_pretrained(
    "models/GLM-4.7-Flash-Marlin-MMFP4", trust_remote_code=True
)
tokens = tokenizer(text, return_tensors="pt").input_ids.to("mps")

# Warmup run
with torch.no_grad():
    _ = model(tokens)
torch.mps.synchronize()

# Profile forward
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.MPS,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    with torch.no_grad():
        _ = model(tokens)
    torch.mps.synchronize()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
