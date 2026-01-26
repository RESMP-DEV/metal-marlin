import time

from metal_marlin.inference.pipeline import MarlinPipeline

models = [
    ("benchmarks/results/qwen3_4b_fp4", "Qwen3-4B FP4"),
]

for path, name in models:
    try:
        pipe = MarlinPipeline.from_pretrained(path, device="mps")

        # Check if using fused kernels
        uses_fused = getattr(pipe.model, "_use_fused_kernels", False)

        # Warmup
        pipe("Hello", max_tokens=5)

        # Benchmark decode
        prompt = "Write a detailed essay:"
        start = time.perf_counter()
        result = pipe(prompt, max_tokens=100, temperature=0.0)
        elapsed = time.perf_counter() - start

        tokens = len(pipe.tokenizer.encode(result)) - len(pipe.tokenizer.encode(prompt))
        tps = tokens / elapsed

        print(f"{name}:")
        print(f"  Fused kernels: {uses_fused}")
        print(f"  Throughput: {tps:.1f} tok/s")
        print()
    except Exception as e:
        print(f"{name}: FAILED - {e}")
