#!/usr/bin/env python3
"""Launch the Qwen3.5-35B-A3B OpenAI-compatible server.

This wrapper uses `metal_marlin.serving.server.run_server()` with Qwen3.5-35B-A3B
defaults and wires serving to `MMFP4Pipeline` (which uses the Trellis model
for hybrid DeltaNet/attention + MoE architecture).

Qwen3.5-35B-A3B Architecture:
    - 256 routed experts, top-8 active (~3.5B active params)
    - Hybrid attention: DeltaNet linear attention + full GQA
    - MLA (Multi-Head Latent Attention) with compressed KV
    - Vocab size: 248,320

Examples:
    QWEN35_MODEL_PATH=models/Qwen3.5-35B-A3B-MMFP4 \
    uv run python scripts/serve_qwen35.py

    uv run python scripts/serve_qwen35.py \
      --model-path models/Qwen3.5-35B-A3B-MMFP4 \
      --port 8000 --batch-size 32

Test:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"qwen3.5-35b-a3b","messages":[{"role":"user","content":"Hello"}]}'
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {value!r}") from exc


@dataclass(frozen=True)
class Qwen35ServeConfig:
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "mps"
    batch_size: int = 32
    enable_batching: bool = True
    use_paged_attention: bool = True
    num_kv_blocks: int = 4096
    block_size: int = 16
    metrics_port: int | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Qwen3.5-35B-A3B OpenAI server with MMFP4 + PagedAttention defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("QWEN35_MODEL_PATH"),
        help="Path to the Qwen3.5-35B-A3B model directory (env: QWEN35_MODEL_PATH).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("QWEN35_PORT", 8000),
        help="Server port (env: QWEN35_PORT).",
    )
    parser.add_argument("--device", default="mps", help="Device to run on (mps/cpu).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_int("QWEN35_BATCH_SIZE", 32),
        help="Maximum concurrent requests (env: QWEN35_BATCH_SIZE).",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=4096,
        help="KV cache blocks for continuous batching.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per KV cache block.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Optional dedicated Prometheus metrics port.",
    )
    parser.add_argument(
        "--no-paged-attention",
        action="store_true",
        help="Disable paged attention toggle on MMFP4 attention layers.",
    )
    return parser


def _install_qwen35_adapter(use_paged_attention: bool) -> None:
    """Make serving load MMFP4Pipeline for Qwen3.5 while keeping serving's call contract."""
    import metal_marlin.serving.engine as serving_engine
    from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

    class Qwen35ServingPipeline:
        def __init__(self, pipeline: MMFP4Pipeline) -> None:
            self._pipeline = pipeline
            self.tokenizer = pipeline.tokenizer
            self.model = pipeline.model
            self.config = getattr(pipeline.model, "config", None)
            if use_paged_attention:
                self._enable_paged_attention()

        @classmethod
        def from_pretrained(cls, model_path: str, device: str = "mps") -> Qwen35ServingPipeline:
            pipeline = MMFP4Pipeline.from_pretrained(
                model_path,
                device=device,
                enable_persistent_cache=True,
                use_paged_attention=use_paged_attention,
            )
            return cls(pipeline)

        def _enable_paged_attention(self) -> None:
            model = getattr(self._pipeline, "model", None)
            model_stack = getattr(model, "model", model)
            layers = getattr(model_stack, "layers", [])
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "use_paged_attention"):
                    setattr(attn, "use_paged_attention", True)

        def __call__(
            self,
            prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.9,
            stream: bool = False,
        ) -> str | Iterator[str]:
            return self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

    serving_engine.MarlinPipeline = Qwen35ServingPipeline  # type: ignore[assignment]


def _parse_config(argv: list[str] | None = None) -> Qwen35ServeConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.model_path:
        parser.error("Missing model path. Provide --model-path or set QWEN35_MODEL_PATH.")
    if args.port <= 0:
        parser.error("--port must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.num_kv_blocks <= 0:
        parser.error("--num-kv-blocks must be > 0.")
    if args.block_size <= 0:
        parser.error("--block-size must be > 0.")

    return Qwen35ServeConfig(
        model_path=str(args.model_path),
        host=str(args.host),
        port=int(args.port),
        device=str(args.device),
        batch_size=int(args.batch_size),
        enable_batching=True,
        use_paged_attention=not bool(args.no_paged_attention),
        num_kv_blocks=int(args.num_kv_blocks),
        block_size=int(args.block_size),
        metrics_port=args.metrics_port,
    )


def main(argv: list[str] | None = None) -> int:
    config = _parse_config(argv)
    _install_qwen35_adapter(use_paged_attention=config.use_paged_attention)

    from metal_marlin.serving.server import run_server

    run_server(
        model_path=config.model_path,
        host=config.host,
        port=config.port,
        device=config.device,
        batch_size=config.batch_size,
        enable_batching=config.enable_batching,
        num_kv_blocks=config.num_kv_blocks,
        block_size=config.block_size,
        metrics_port=config.metrics_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
