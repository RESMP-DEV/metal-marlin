import time

from metal_marlin.inference.pipeline import MarlinPipeline


def benchmark_glm4():
    pipe = MarlinPipeline.from_pretrained(
        "benchmarks/results/glm47_sensitivity_fp8_int2",
        device="mps",
    )

    # Warmup
    pipe("Hello", max_tokens=5)

    # Benchmark
    prompt = "Write a detailed essay about quantum computing:"
    start = time.perf_counter()
    result = pipe(prompt, max_tokens=100, temperature=0.0)
    elapsed = time.perf_counter() - start

    tokens = len(pipe.tokenizer.encode(result)) - len(pipe.tokenizer.encode(prompt))
    print("GLM-4.7-Flash FP8+INT2 MoE")
    print(f"  Tokens: {tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {tokens/elapsed:.1f} tok/s")

    return {"model": "GLM-4.7-Flash", "tok_s": tokens / elapsed}


if __name__ == "__main__":
    benchmark_glm4()
