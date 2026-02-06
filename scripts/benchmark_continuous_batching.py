#!/usr/bin/env python3
"""Benchmark continuous batching throughput.

This script benchmarks throughput with two modes:

1. Batch Size Mode (--mode=batch, default):
   Compares single-request processing (batch=1) vs continuous batching
   with batch sizes 4, 8, 16.

   Expected result: 5-10x throughput improvement with batching.

   Usage:
       uv run python scripts/benchmark_continuous_batching.py --mock

2. Context Length Mode (--mode=context or --context-lengths):
   Benchmarks Requests/second vs Context Length.

   Shows how throughput degrades with longer prompts due to:
   - Larger prefill cost (O(seq_len))
   - More KV cache memory usage
   - Higher memory bandwidth requirements

   Usage:
       uv run python scripts/benchmark_continuous_batching.py --mock --mode=context \
           --context-lengths 64 128 256 512 1024

Usage Examples:
    # Batch size benchmark with real model
    uv run python scripts/benchmark_continuous_batching.py --model /path/to/model

    # Quick test with mock model
    uv run python scripts/benchmark_continuous_batching.py --mock

    # Context length benchmark
    uv run python scripts/benchmark_continuous_batching.py --mock --mode context \
        --context-lengths 64 128 256 512 1024

    # Custom batch sizes
    uv run python scripts/benchmark_continuous_batching.py --mock --batch-sizes 4 8 16
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    mode: str
    batch_size: int
    num_requests: int
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    requests_per_second: float
    avg_latency_ms: float
    context_length: int = 0  # Number of prompt tokens


def benchmark_single_request(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> BenchmarkResult:
    """Benchmark single-request processing (batch=1)."""
    import torch

    from metal_marlin.kv_cache import TrellisKVCache

    device = next(model.parameters()).device
    total_tokens = 0
    start_time = time.time()

    for prompt in prompts:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device=device,
        )

        # Create KV cache
        kv_cache = TrellisKVCache(
            num_layers=model.config.num_hidden_layers,
            batch_size=1,
            max_seq_len=input_ids.shape[1] + max_new_tokens,
            kv_lora_rank=model.config.kv_lora_rank,
            qk_rope_head_dim=model.config.qk_rope_head_dim,
            device=str(device),
        )

        # Prefill
        with torch.inference_mode():
            output = model.forward(input_ids, kv_cache=kv_cache)

        # Decode
        generated = 0
        for _ in range(max_new_tokens):
            next_logits = output.logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            with torch.inference_mode():
                output = model.forward(next_token, kv_cache=kv_cache)

            generated += 1

        total_tokens += generated

    elapsed = time.time() - start_time

    return BenchmarkResult(
        mode="single",
        batch_size=1,
        num_requests=len(prompts),
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=total_tokens / elapsed,
        requests_per_second=len(prompts) / elapsed,
        avg_latency_ms=(elapsed / len(prompts)) * 1000,
    )


def benchmark_continuous_batching(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> BenchmarkResult:
    """Benchmark continuous batching with given batch size."""
    from metal_marlin.serving.continuous_engine import (
        ContinuousBatchingEngine,
        EngineConfig,
        GenerationConfig,
    )

    config = EngineConfig(
        max_batch_size=batch_size,
        max_seq_len=2048,
    )

    engine = ContinuousBatchingEngine(model, tokenizer, config)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=0,  # Greedy for reproducibility
        top_p=1.0,
    )

    start_time = time.time()

    # Submit all requests
    futures = engine.submit_batch(prompts, gen_config)

    # Run until all complete
    engine.run_until_empty()

    # Get results
    results = [f.result() for f in futures]

    elapsed = time.time() - start_time

    stats = engine.get_stats()
    total_tokens = stats["total_tokens"]

    return BenchmarkResult(
        mode="continuous",
        batch_size=batch_size,
        num_requests=len(prompts),
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
        requests_per_second=len(prompts) / elapsed if elapsed > 0 else 0,
        avg_latency_ms=(elapsed / len(prompts)) * 1000 if prompts else 0,
    )


def create_mock_model():
    """Create a mock model for testing without a real model.

    The mock simulates GPU behavior where:
    - There's a fixed kernel launch overhead per forward pass (~5ms)
    - Actual compute scales sublinearly with batch size (parallelism)
    - This makes batching beneficial: 16 requests at once is faster than 16x1
    """
    import torch
    import torch.nn as nn

    class MockConfig:
        num_hidden_layers = 4
        kv_lora_rank = 64
        qk_rope_head_dim = 16
        vocab_size = 1000

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self._dummy = nn.Linear(1, 1)  # Just for parameters()

        def forward(self, input_ids, attention_mask=None, kv_cache=None, position_ids=None):
            batch_size, seq_len = input_ids.shape

            # Simulate GPU behavior:
            # - Fixed kernel launch overhead: ~2ms
            # - Compute scales with sqrt(batch_size) due to parallelism
            # This makes batching beneficial
            base_overhead_ms = 2.0  # Fixed overhead per forward pass
            compute_per_item_ms = 0.5  # Compute per batch item (sublinear due to GPU parallelism)

            # Total time = overhead + sqrt(batch_size) * compute_per_item
            # This simulates that batch=16 takes ~4x the compute of batch=1,
            # not 16x, making batching worthwhile
            import math
            total_ms = base_overhead_ms + math.sqrt(batch_size) * compute_per_item_ms
            time.sleep(total_ms / 1000)

            # Return mock logits
            logits = torch.randn(batch_size, seq_len, self.config.vocab_size, device=input_ids.device)

            class Output:
                pass

            out = Output()
            out.logits = logits
            return out

    return MockModel().to("mps" if torch.backends.mps.is_available() else "cpu")


def benchmark_context_lengths(
    model,
    tokenizer,
    max_new_tokens: int,
    context_lengths: list[int],
    batch_size: int = 4,
    num_requests_per_length: int = 16,
) -> list[BenchmarkResult]:
    """Benchmark throughput (Req/sec) vs Context Length.

    This simulates realistic server load where requests arrive with varying
    prompt lengths. Results show how throughput degrades with longer
    context lengths due to:
    - Larger prefill cost (O(seq_len))
    - More KV cache memory usage
    - Higher memory bandwidth requirements

    Args:
        model: The language model.
        tokenizer: Tokenizer for encoding/decoding.
        max_new_tokens: Max output tokens per request.
        context_lengths: List of context lengths to test.
        batch_size: Batch size for continuous batching.
        num_requests_per_length: Number of requests per context length.

    Returns:
        List of BenchmarkResult with context_length set.
    """
    import torch

    from metal_marlin.serving.request import GenerationRequest, RequestStatus
    from metal_marlin.serving.scheduler import (
        BatchScheduler,
        SchedulerConfig,
    )
    from metal_marlin.kv_cache import TrellisKVCache

    device = next(model.parameters()).device
    results: list[BenchmarkResult] = []

    # Setup scheduler and allocator
    from metal_marlin.paged.allocator import BlockAllocator

    config = SchedulerConfig(
        max_num_seqs=batch_size * 2,  # Allow some waiting
        max_num_batched_tokens=4096,
        max_prefill_tokens=2048,
        block_size=16,
    )
    allocator = BlockAllocator(num_blocks=1024)
    scheduler = BatchScheduler(config, allocator)

    for ctx_len in context_lengths:
        # Create prompts of varying lengths
        prompts = []
        base_text = "This is a test prompt. " * (ctx_len // 10 + 1)
        for i in range(num_requests_per_length):
            # Create prompt with roughly ctx_len tokens
            repeat = max(1, ctx_len // 10)
            prompt_text = " ".join([f"word{j}" for j in range(repeat)])
            prompts.append(prompt_text)

        # Tokenize all prompts
        tokenized_prompts = [
            tokenizer.encode(p)[:ctx_len] for p in prompts
        ]

        # Create KV cache
        max_seq = ctx_len + max_new_tokens
        kv_cache = TrellisKVCache(
            num_layers=model.config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=max_seq,
            kv_lora_rank=model.config.kv_lora_rank,
            qk_rope_head_dim=model.config.qk_rope_head_dim,
            device=str(device),
        )

        # Create generation requests
        requests = [
            GenerationRequest(
                request_id=f"req-{i}",
                prompt_tokens=tokens,
                max_tokens=max_new_tokens,
                temperature=1.0,
            )
            for i, tokens in enumerate(tokenized_prompts)
        ]

        # Benchmark
        total_tokens = 0
        start_time = time.time()

        # Submit all requests
        scheduler.insert_batch(requests)

        # Run generation loop
        iteration = 0
        while scheduler.has_pending_work:
            # Get schedule
            output = scheduler.schedule()

            if not (output.prefill_requests or output.decode_requests):
                break

            # Prepare batch
            prefill_batch = []
            decode_batch = []
            prefill_seqs = []

            for req in output.prefill_requests:
                input_ids = torch.tensor(
                    [req.prompt_tokens],
                    dtype=torch.long,
                    device=device,
                )
                prefill_batch.append(input_ids)
                prefill_seqs.append(req)

            for req in output.decode_requests:
                last_token = torch.tensor(
                    [[req.output_tokens[-1]]],
                    dtype=torch.long,
                    device=device,
                )
                decode_batch.append(last_token)

            # Run forward pass
            with torch.inference_mode():
                if prefill_batch:
                    # Stack prefill batch
                    stacked_prefill = torch.cat(prefill_batch, dim=0)

                    # Run prefill for each request (simplified)
                    for input_ids in prefill_batch:
                        _ = model.forward(input_ids, kv_cache=kv_cache)

                        # Mark as done with prefill
                        for req in prefill_seqs:
                            req.status = RequestStatus.RUNNING

                if decode_batch:
                    # Stack decode batch
                    stacked_decode = torch.cat(decode_batch, dim=0)

                    # Run decode
                    output_tensor = model.forward(stacked_decode, kv_cache=kv_cache)

                    # Sample tokens
                    for i, req in enumerate(output.decode_requests):
                        next_logits = output_tensor.logits[i, -1, :]
                        next_token = int(torch.argmax(next_logits, dim=-1).item())

                        if not req.output_tokens:
                            req.first_token_time = time()

                        req.output_tokens.append(next_token)
                        total_tokens += 1

                        # Update request status
                        if req.is_finished:
                            req.status = RequestStatus.FINISHED
                            req.completion_time = time()

            # Free finished requests
            scheduler.free_finished()

            # Step decode sequences
            scheduler.step_decode()

            iteration += 1

        elapsed = time.time() - start_time

        result = BenchmarkResult(
            mode="context_length",
            batch_size=batch_size,
            num_requests=num_requests_per_length,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
            requests_per_second=num_requests_per_length / elapsed if elapsed > 0 else 0,
            avg_latency_ms=(elapsed / num_requests_per_length) * 1000 if num_requests_per_length > 0 else 0,
            context_length=ctx_len,
        )
        results.append(result)

        # Reset for next context length
        scheduler = BatchScheduler(config, BlockAllocator(num_blocks=1024))
        kv_cache.reset()

    return results


class MockTokenizer:
    """Mock tokenizer for testing."""

    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        # Simple word-based tokenization
        return [hash(w) % 1000 for w in text.split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"tok{t}" for t in token_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark continuous batching throughput"
    )
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--mock", action="store_true", help="Use mock model for testing")
    parser.add_argument("--num-requests", type=int, default=32, help="Number of requests")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens per request")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Batch sizes to test",
    )
    parser.add_argument("--skip-single", action="store_true", help="Skip single-request benchmark")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to test (enables context length benchmarking mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "context", "both"],
        default="batch",
        help="Benchmark mode: batch (vs batch size), context (vs context length), both",
    )
    parser.add_argument(
        "--req-per-length",
        type=int,
        default=16,
        help="Requests per context length when testing context lengths",
    )
    args = parser.parse_args()

    # Load or create model
    if args.mock:
        print("Using mock model for testing...")
        model = create_mock_model()
        tokenizer = MockTokenizer()
    elif args.model:
        print(f"Loading model from {args.model}...")
        from transformers import AutoTokenizer

        from metal_marlin.trellis.model import TrellisForCausalLM

        model = TrellisForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        print("Error: Must specify --model or --mock")
        return 1

    # Create test prompts
    prompts = [
        f"Hello, this is test prompt number {i}. Please tell me about"
        for i in range(args.num_requests)
    ]

    results: list[BenchmarkResult] = []

    # Determine which benchmark to run
    if args.mode == "context" or args.context_lengths:
        # Context length benchmarking mode
        ctx_lengths = args.context_lengths or [64, 128, 256, 512, 1024]
        batch_size = args.batch_sizes[0] if args.batch_sizes else 4

        print("\nContext Length Benchmark:")
        print("=" * 50)
        print(f"  Batch size: {batch_size}")
        print(f"  Requests per length: {args.req_per_length}")
        print(f"  Max output tokens: {args.max_tokens}")
        print(f"  Context lengths: {ctx_lengths}")
        print()

        context_results = benchmark_context_lengths(
            model,
            tokenizer,
            args.max_tokens,
            ctx_lengths,
            batch_size=batch_size,
            num_requests_per_length=args.req_per_length,
        )
        results.extend(context_results)

        # Context length summary
        print("\n" + "=" * 70)
        print("CONTEXT LENGTH BENCHMARK RESULTS")
        print("=" * 70)
        print(
            f"{'CtxLen':>8} {'Tok/s':>12} {'Req/s':>12} {'Latency':>12} {'Rel. Thru':>12}"
        )
        print("-" * 70)

        baseline_tps = context_results[0].tokens_per_second if context_results else 1

        for r in context_results:
            relative = (r.tokens_per_second / baseline_tps) if baseline_tps > 0 else 0
            print(
                f"{r.context_length:>8} {r.tokens_per_second:>12.1f} "
                f"{r.requests_per_second:>12.2f} {r.avg_latency_ms:>11.1f}ms {relative:>11.1f}x"
            )

        print("=" * 70)

    else:
        # Batch size benchmarking mode (default)
        print("\nBatch Size Benchmark:")
        print("=" * 50)
        print(f"  Requests: {args.num_requests}")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Batch sizes: {args.batch_sizes}")
        print()

        # Benchmark single-request (baseline)
        if not args.skip_single:
            print("Running single-request benchmark (baseline)...")
            result = benchmark_single_request(model, tokenizer, prompts, args.max_tokens)
            results.append(result)
            print(f"  Tokens/sec: {result.tokens_per_second:.1f}")
            print(f"  Requests/sec: {result.requests_per_second:.2f}")
            print()

        # Benchmark continuous batching with different batch sizes
        for batch_size in args.batch_sizes:
            print(f"Running continuous batching benchmark (batch_size={batch_size})...")
            result = benchmark_continuous_batching(
                model, tokenizer, prompts, args.max_tokens, batch_size
            )
            results.append(result)
            print(f"  Tokens/sec: {result.tokens_per_second:.1f}")
            print(f"  Requests/sec: {result.requests_per_second:.2f}")
            print()

        # Batch size summary
        print("\n" + "=" * 70)
        print("BATCH SIZE BENCHMARK RESULTS")
        print("=" * 70)
        print(
            f"{'Mode':<15} {'Batch':>6} {'Tok/s':>10} {'Req/s':>10} {'Latency':>12} {'Speedup':>10}"
        )
        print("-" * 70)

        baseline_tps = results[0].tokens_per_second if results else 1

        for r in results:
            speedup = r.tokens_per_second / baseline_tps if baseline_tps > 0 else 0
            print(
                f"{r.mode:<15} {r.batch_size:>6} {r.tokens_per_second:>10.1f} "
                f"{r.requests_per_second:>10.2f} {r.avg_latency_ms:>10.1f}ms {speedup:>9.1f}x"
            )

        print("=" * 70)

        # Check if we achieved expected speedup
        if len(results) > 1:
            max_speedup = max(r.tokens_per_second for r in results[1:]) / baseline_tps
            print(f"\nMax speedup achieved: {max_speedup:.1f}x")
            if max_speedup >= 3:
                print("✓ Significant throughput improvement achieved!")
            else:
                print("! Lower than expected speedup - may need optimization")

    return 0


if __name__ == "__main__":
    exit(main())
