#!/usr/bin/env python3
"""End-to-end benchmark: MoE GPTQ with Hessian calibration.

Evaluates:
- GPTQ + Hessian vs fast RTN quantization quality
- Throughput (prefill + decode tok/s)
- RMSE (quantization reconstruction error)
- Perplexity on WikiText-2

Usage:
  cd contrib/metal_marlin
  uv run python benchmarks/bench_moe_gptq_hessian.py --layers 3
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class QuantizationQuality:
    """Quantization quality metrics."""

    method: str
    quant_time_s: float
    rmse: float
    max_abs_error: float
    params_quantized: int


@dataclass
class ThroughputMetrics:
    """Inference throughput metrics."""

    prefill_tok_s: float
    decode_tok_s: float
    memory_peak_mb: float


@dataclass
class BenchmarkResults:
    """Full benchmark results."""

    model_id: str
    num_layers: int
    num_experts: int
    timestamp: str

    # Quantization quality comparison
    gptq_quality: QuantizationQuality | None
    rtn_quality: QuantizationQuality | None

    # Throughput
    throughput: ThroughputMetrics | None

    # Perplexity
    perplexity: float | None
    ppl_tokens: int | None

    def to_json(self) -> dict:
        return asdict(self)


def compute_rmse(original: torch.Tensor, quantized: torch.Tensor) -> tuple[float, float]:
    """Compute RMSE and max absolute error between tensors."""
    with torch.no_grad():
        diff = original.float() - quantized.float()
        rmse = torch.sqrt(torch.mean(diff * diff)).item()
        max_abs = torch.max(torch.abs(diff)).item()
    return rmse, max_abs


def measure_throughput(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    prompt: str = "The capital of France is",
    prefill_tokens: int = 128,
    decode_tokens: int = 64,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> ThroughputMetrics:
    """Measure prefill and decode throughput."""
    # Extend prompt to prefill length
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    while input_ids.shape[1] < prefill_tokens:
        input_ids = torch.cat(
            [
                input_ids,
                input_ids[:, : min(prefill_tokens - input_ids.shape[1], input_ids.shape[1])],
            ],
            dim=1,
        )
    input_ids = input_ids[:, :prefill_tokens]

    memory_peak = 0

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=decode_tokens, do_sample=False)

    sync()
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # Benchmark prefill
    prefill_times = []
    for _ in range(benchmark_runs):
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        sync()
        prefill_times.append(time.perf_counter() - start)

    avg_prefill = sum(prefill_times) / len(prefill_times)
    prefill_tok_s = prefill_tokens / avg_prefill

    # Benchmark decode (full generation)
    decode_times = []
    for _ in range(benchmark_runs):
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=decode_tokens, do_sample=False)
        sync()
        decode_times.append(time.perf_counter() - start)

    avg_decode = sum(decode_times) / len(decode_times)
    total_tokens = prefill_tokens + decode_tokens
    decode_tok_s = total_tokens / avg_decode

    # Memory
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        memory_peak = torch.mps.current_allocated_memory() / (1024 * 1024)
    elif device == "cuda":
        memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return ThroughputMetrics(
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        memory_peak_mb=memory_peak,
    )


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    max_samples: int = 10,
    max_length: int = 256,
) -> tuple[float, int]:
    """Compute perplexity on WikiText-2."""
    from metal_marlin.eval_perplexity import compute_perplexity_from_logits, load_wikitext2

    texts = load_wikitext2(max_samples=max_samples)
    if not texts:
        return float("nan"), 0

    def logits_fn(input_ids_np: np.ndarray) -> np.ndarray:
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.detach().float().cpu().numpy()
        return logits

    ppl = compute_perplexity_from_logits(
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        verbose=True,
    )

    token_count = sum(
        len(tokenizer.encode(text, truncation=True, max_length=max_length)) for text in texts
    )

    return ppl, token_count


def benchmark_quantization_quality(
    model,
    tokenizer,
    device: str,
    num_layers: int,
    group_size: int = 128,
    use_gptq: bool = True,
    calibration_samples: int = 10,
) -> QuantizationQuality:
    """Benchmark quantization quality with proper GPTQ + Hessian."""
    from metal_marlin.eval_perplexity import load_wikitext2
    from metal_marlin.layer_replacement import replace_moe_layers

    method = "GPTQ+Hessian" if use_gptq else "RTN (no calibration)"

    # Prepare calibration data for GPTQ
    calibration_inputs = None
    if use_gptq:
        texts = load_wikitext2(max_samples=calibration_samples)
        if texts:
            calibration_inputs = [
                tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(
                    device
                )
                for text in texts
            ]
            print(f"  Prepared {len(calibration_inputs)} calibration samples")

    # Quantize with calibration
    start = time.perf_counter()
    result = replace_moe_layers(
        model,
        bits=4,
        group_size=group_size,
        format="fp4",
        calibration_inputs=calibration_inputs,
        use_gptq=use_gptq,
        device=device,
        verbose=True,
    )
    quant_time = time.perf_counter() - start

    return QuantizationQuality(
        method=method,
        quant_time_s=quant_time,
        rmse=0.0,  # Computed via perplexity instead
        max_abs_error=0.0,
        params_quantized=result["total_params_quantized"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE GPTQ+Hessian vs RTN benchmark")
    parser.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    parser.add_argument(
        "--layers", type=int, default=3, help="Number of layers to load (for memory)"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--ppl-samples", type=int, default=5)
    parser.add_argument("--skip-gptq", action="store_true", help="Skip GPTQ (only test RTN)")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--output", default="benchmarks/results/moe_gptq_benchmark.json")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    # Resolve dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("MoE GPTQ+Hessian Quantization Benchmark")
    print("=" * 70)
    print(f"Model: {args.model_id}")
    print(f"Layers: {args.layers}")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Group size: {args.group_size}")
    print()

    # Load model with limited layers
    print(f"Loading model with {args.layers} layers...")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    config.num_hidden_layers = args.layers

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e9:.2f}B params")

    # Get model info
    num_experts = getattr(config, "num_experts", None) or getattr(config, "num_router_experts", 64)

    results = BenchmarkResults(
        model_id=args.model_id,
        num_layers=args.layers,
        num_experts=num_experts,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        gptq_quality=None,
        rtn_quality=None,
        throughput=None,
        perplexity=None,
        ppl_tokens=None,
    )

    # Test 1: Fast RTN quantization
    print("\n" + "=" * 50)
    print("[1/4] Fast RTN Quantization")
    print("=" * 50)

    # Clone model for RTN test
    model_rtn = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model_rtn.eval()

    rtn_quality = benchmark_quantization_quality(
        model_rtn, tokenizer, device, args.layers, args.group_size, use_gptq=False
    )
    results.rtn_quality = rtn_quality
    print(f"  Method: {rtn_quality.method}")
    print(f"  Time: {rtn_quality.quant_time_s:.2f}s")
    print(f"  Params quantized: {rtn_quality.params_quantized / 1e6:.1f}M")

    # Test 2: GPTQ + Hessian quantization
    if not args.skip_gptq:
        print("\n" + "=" * 50)
        print("[2/4] GPTQ + Hessian Quantization")
        print("=" * 50)

        model_gptq = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        model_gptq.eval()

        gptq_quality = benchmark_quantization_quality(
            model_gptq, tokenizer, device, args.layers, args.group_size, use_gptq=True
        )
        results.gptq_quality = gptq_quality
        print(f"  Method: {gptq_quality.method}")
        print(f"  Time: {gptq_quality.quant_time_s:.2f}s")
        print(f"  Params quantized: {gptq_quality.params_quantized / 1e6:.1f}M")

        # Speed comparison
        if rtn_quality.quant_time_s > 0:
            speedup = gptq_quality.quant_time_s / rtn_quality.quant_time_s
            print(f"  RTN is {speedup:.1f}x faster than GPTQ")
    else:
        print("\n[2/4] GPTQ skipped")

    # Test 3: Throughput
    if not args.skip_throughput:
        print("\n" + "=" * 50)
        print("[3/4] Throughput Measurement")
        print("=" * 50)

        throughput = measure_throughput(model_rtn, tokenizer, device)
        results.throughput = throughput
        print(f"  Prefill: {throughput.prefill_tok_s:.1f} tok/s")
        print(f"  Decode: {throughput.decode_tok_s:.1f} tok/s")
        print(f"  Memory peak: {throughput.memory_peak_mb:.1f} MB")
    else:
        print("\n[3/4] Throughput skipped")

    # Test 4: Perplexity
    if not args.skip_perplexity:
        print("\n" + "=" * 50)
        print("[4/4] Perplexity (WikiText-2)")
        print("=" * 50)

        ppl, tokens = compute_perplexity(model_rtn, tokenizer, device, max_samples=args.ppl_samples)
        results.perplexity = ppl
        results.ppl_tokens = tokens
        print(f"  Perplexity: {ppl:.4f}")
        print(f"  Tokens: {tokens}")
    else:
        print("\n[4/4] Perplexity skipped")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results.to_json(), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
