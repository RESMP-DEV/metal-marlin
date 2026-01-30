import time

from metal_marlin.inference.pipeline import MarlinPipeline

model_path = "benchmarks/results/qwen3_4b_fp4"

try:
    # Test with fused kernels
    pipe_fused = MarlinPipeline.from_pretrained(model_path, device="mps")
    pipe_fused.model._use_fused_kernels = True

    # Test with fallback
    pipe_fallback = MarlinPipeline.from_pretrained(model_path, device="mps")
    pipe_fallback.model._use_fused_kernels = False

    prompt = "The capital of France is"

    for name, pipe in [("Fused", pipe_fused), ("Fallback", pipe_fallback)]:
        pipe(prompt, max_tokens=5)  # Warmup

        start = time.perf_counter()
        result = pipe(prompt, max_tokens=50, temperature=0.0)
        elapsed = time.perf_counter() - start

        tokens = len(pipe.tokenizer.encode(result)) - len(
            pipe.tokenizer.encode(prompt)
        )
        print(f"{name}: {tokens/elapsed:.1f} tok/s")
except Exception as e:
    print(f"Benchmark failed: {e}")
