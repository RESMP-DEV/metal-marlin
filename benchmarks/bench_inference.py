"""
End-to-end inference benchmarks.

Benchmarks prefill and decode performance of a synthetic Llama-like model
built from MarlinLinear (FP4-quantized GEMM) layers. Measures tok/s for
both phases independently and compares quantization levels.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Check MLX availability early for clear error message
try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from metal_marlin import MarlinLinear  # noqa: E402


@dataclass
class LlamaConfig:
    """Synthetic Llama model configuration for benchmarking."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    max_seq_len: int = 2048
    quant_type: str = "fp4"  # "fp4" or "none"
    group_size: int = 32

    @classmethod
    def llama_7b(cls) -> LlamaConfig:
        return cls()

    @classmethod
    def llama_3b(cls) -> LlamaConfig:
        return cls(
            hidden_size=3200,
            intermediate_size=8640,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=100,
        )

    @classmethod
    def tiny(cls) -> LlamaConfig:
        """Small config for quick smoke-test benchmarks."""
        return cls(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=128,
        )


class SyntheticLlamaLayer(nn.Module):
    """Single transformer layer using MarlinLinear for weight-heavy ops."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        h = config.hidden_size
        kv_dim = config.num_key_value_heads * config.head_dim
        ffn = config.intermediate_size

        if config.quant_type == "fp4":
            self.q_proj = self._make_marlin(h, h, config.group_size)
            self.k_proj = self._make_marlin(h, kv_dim, config.group_size)
            self.v_proj = self._make_marlin(h, kv_dim, config.group_size)
            self.o_proj = self._make_marlin(h, h, config.group_size)
            self.gate_proj = self._make_marlin(h, ffn, config.group_size)
            self.up_proj = self._make_marlin(h, ffn, config.group_size)
            self.down_proj = self._make_marlin(ffn, h, config.group_size)
        else:
            self.q_proj = nn.Linear(h, h, bias=False)
            self.k_proj = nn.Linear(h, kv_dim, bias=False)
            self.v_proj = nn.Linear(h, kv_dim, bias=False)
            self.o_proj = nn.Linear(h, h, bias=False)
            self.gate_proj = nn.Linear(h, ffn, bias=False)
            self.up_proj = nn.Linear(h, ffn, bias=False)
            self.down_proj = nn.Linear(ffn, h, bias=False)

        self.input_layernorm = nn.RMSNorm(h)
        self.post_attention_layernorm = nn.RMSNorm(h)
        self.config = config

    @staticmethod
    def _make_marlin(in_f: int, out_f: int, group_size: int) -> MarlinLinear:
        return MarlinLinear(
            in_features=in_f,
            out_features=out_f,
            bias=False,
            quant_type="fp4",
            group_size=group_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass simulating attention + FFN (no actual attention compute)."""
        # Attention block (projections only, no softmax/cache for raw throughput)
        residual = x
        h = self.input_layernorm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # Simulate attention output as sum (avoids softmax overhead in GEMM bench)
        attn_out = self.o_proj(q + k + v)
        h = residual + attn_out

        # FFN block (SwiGLU)
        residual = h
        h = self.post_attention_layernorm(h)
        gate = self.gate_proj(h)
        up = self.up_proj(h)
        h = self.down_proj(nn.silu(gate) * up)
        return residual + h


class SyntheticLlama(nn.Module):
    """Minimal Llama-like model for benchmarking GEMM throughput."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [SyntheticLlamaLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size)
        # LM head as standard linear (not quantized, matches real models)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Full forward pass: embed -> layers -> norm -> lm_head."""
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


@dataclass
class InferenceMetrics:
    prompt_tokens: int
    generated_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    overall_tok_s: float


def benchmark_inference(
    model: SyntheticLlama,
    prompt_lengths: list[int] | None = None,
    gen_lengths: list[int] | None = None,
    warmup: int = 2,
    iterations: int = 5,
) -> list[InferenceMetrics]:
    """Benchmark prefill and decode speeds.

    Prefill: full forward pass on prompt_len tokens.
    Decode: sequential single-token forward passes for gen_len steps.
    """
    if prompt_lengths is None:
        prompt_lengths = [32, 128, 512]
    if gen_lengths is None:
        gen_lengths = [32, 128, 256]

    results: list[InferenceMetrics] = []

    for prompt_len in prompt_lengths:
        for gen_len in gen_lengths:
            input_ids = mx.ones((1, prompt_len), dtype=mx.int32)

            # Warmup
            for _ in range(warmup):
                _ = model(input_ids)
                mx.synchronize()

            prefill_times: list[float] = []
            decode_times: list[float] = []

            for _ in range(iterations):
                # Prefill
                start = time.perf_counter()
                logits = model(input_ids)
                mx.eval(logits)
                mx.synchronize()
                prefill_times.append(time.perf_counter() - start)

                # Decode: sequential single-token passes
                start = time.perf_counter()
                for _ in range(gen_len):
                    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
                    logits = model(next_token)
                    mx.eval(logits)
                mx.synchronize()
                decode_times.append(time.perf_counter() - start)

            avg_prefill = sum(prefill_times) / len(prefill_times) * 1000
            avg_decode = sum(decode_times) / len(decode_times) * 1000

            metrics = InferenceMetrics(
                prompt_tokens=prompt_len,
                generated_tokens=gen_len,
                prefill_time_ms=avg_prefill,
                decode_time_ms=avg_decode,
                total_time_ms=avg_prefill + avg_decode,
                prefill_tok_s=prompt_len / (avg_prefill / 1000),
                decode_tok_s=gen_len / (avg_decode / 1000),
                overall_tok_s=(prompt_len + gen_len) / ((avg_prefill + avg_decode) / 1000),
            )
            results.append(metrics)

            print(
                f"Prompt={prompt_len:>4}, Gen={gen_len:>4}: "
                f"Prefill {metrics.prefill_tok_s:>8.0f} tok/s, "
                f"Decode {metrics.decode_tok_s:>7.1f} tok/s"
            )

    return results


def compare_quantization_levels(
    config: LlamaConfig | None = None,
    prompt_len: int = 128,
    gen_len: int = 128,
    warmup: int = 2,
) -> None:
    """Compare FP16 vs FP4 inference speed on identical architecture."""
    if config is None:
        config = LlamaConfig.tiny()

    input_ids = mx.ones((1, prompt_len), dtype=mx.int32)

    for quant_type, label in [("none", "FP16"), ("fp4", "FP4")]:
        print(f"Building {label} model ({config.num_hidden_layers}L, h={config.hidden_size})...")
        cfg = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            quant_type=quant_type,
            group_size=config.group_size,
        )
        model = SyntheticLlama(cfg)
        mx.eval(model.parameters())

        # Warmup
        for _ in range(warmup):
            _ = model(input_ids)
            mx.synchronize()

        # Prefill
        start = time.perf_counter()
        logits = model(input_ids)
        mx.eval(logits)
        mx.synchronize()
        prefill_ms = (time.perf_counter() - start) * 1000

        # Decode
        start = time.perf_counter()
        for _ in range(gen_len):
            next_token = mx.argmax(logits[:, -1:, :], axis=-1)
            logits = model(next_token)
            mx.eval(logits)
        mx.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000

        total_tokens = prompt_len + gen_len
        total_s = (prefill_ms + decode_ms) / 1000
        print(
            f"  {label}: Prefill {prompt_len / (prefill_ms / 1000):.0f} tok/s, "
            f"Decode {gen_len / (decode_ms / 1000):.1f} tok/s, "
            f"Overall {total_tokens / total_s:.1f} tok/s"
        )

        # Free model memory
        del model
        mx.synchronize()


if __name__ == "__main__":
    if not HAS_MLX:
        print("ERROR: Benchmarks require MLX for Metal GPU access.")
        print("Install with: pip install mlx")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="End-to-end inference benchmark")
    parser.add_argument(
        "--size",
        choices=["tiny", "3b", "7b"],
        default="tiny",
        help="Model size preset (default: tiny for quick test)",
    )
    parser.add_argument("--compare", action="store_true", help="Compare FP16 vs FP4")
    parser.add_argument("--prompt-len", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--gen-len", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    configs = {"tiny": LlamaConfig.tiny, "3b": LlamaConfig.llama_3b, "7b": LlamaConfig.llama_7b}
    config = configs[args.size]()

    if args.compare:
        compare_quantization_levels(config, prompt_len=args.prompt_len[0], gen_len=args.gen_len[0])
    else:
        print(f"Model: {args.size} ({config.num_hidden_layers}L, h={config.hidden_size}, "
              f"quant={config.quant_type})")
        model = SyntheticLlama(config)
        mx.eval(model.parameters())
        benchmark_inference(
            model,
            prompt_lengths=args.prompt_len,
            gen_lengths=args.gen_len,
            warmup=args.warmup,
            iterations=args.iterations,
        )
