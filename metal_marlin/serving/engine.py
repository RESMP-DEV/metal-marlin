from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..inference.pipeline import MarlinPipeline
from .continuous_batch import BatchScheduler, KVCacheManager, SchedulerConfig
from .openai_schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from .request import GenerationRequest, RequestStatus, RequestTimeoutError


def _detect_model_format(model_path: str) -> str:
    """Detect if model is Marlin or Trellis format.

    Returns:
        'trellis' if model uses Trellis quantization
        'marlin' otherwise (default)
    """
    path = Path(model_path)
    if not path.exists():
        return "marlin"  # Will fail later with proper error

    # Check for Trellis markers
    config_path = path / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            # Trellis models have specific markers
            if config.get("quantization_config", {}).get("quant_method") == "trellis":
                return "trellis"
            if config.get("format") == "trellis":
                return "trellis"
        except (json.JSONDecodeError, KeyError):
            pass

    # Check for Trellis directory structure (sharded layers)
    if (path / "layer_0000").exists():
        return "trellis"

    return "marlin"


class _MockTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def encode(self, text: str) -> list[int]:
        # Simple tokenization by whitespace for counting.
        return [idx for idx, _ in enumerate(text.split())]

    def decode(self, tokens: list[int]) -> str:
        # Simple decoding for mock mode.
        return " ".join(f"token{i}" for i in range(len(tokens)))


class _MockPipeline:
    def __init__(self, model_name: str):
        self.tokenizer = _MockTokenizer()
        self.device = "cpu"
        self._model_name = model_name

    def __call__(
        self,
        prompt: str | list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        _ = (max_tokens, temperature, top_p)
        response = " mock response"
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if stream:
            pieces = [piece + " " for piece in response.split()]
            return iter(pieces)
        return f"{prompt}{response}"


@dataclass
class EngineConfig:
    model_path: str
    device: str = "mps"
    max_model_len: int = 4096
    max_batch_size: int = 32
    request_timeout: float = 60.0
    enable_batching: bool = False  # Start disabled for compatibility
    num_kv_blocks: int = 512
    block_size: int = 16


class ServingEngine:
    """Async-compatible inference engine for OpenAI API.

    This implementation bridges the MarlinPipeline to the serving layer and
    provides async request handling with optional streaming.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        use_mock = os.getenv("METAL_MARLIN_MOCK_MODEL") == "1"
        model_name = os.getenv("METAL_MARLIN_MOCK_MODEL_NAME")
        if model_name is None:
            model_name = (
                str(config.model_path).split("/")[-1] if config.model_path else "mock-model"
            )

        model_format = _detect_model_format(config.model_path)
        self._model_format = model_format

        if use_mock:
            self.pipeline = _MockPipeline(model_name)
        elif model_format == "trellis":
            self.pipeline = self._load_trellis_pipeline(config)
        else:
            # Default: Marlin format
            if not Path(config.model_path).exists():
                raise FileNotFoundError(f"Model not found: {config.model_path}")
            try:
                self.pipeline = MarlinPipeline.from_pretrained(
                    config.model_path, device=config.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {config.model_path}: {e}") from e
        self.model_name = model_name
        self._request_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self._results: dict[str, asyncio.Future] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if config.enable_batching:
            self.kv_manager = KVCacheManager(
                num_blocks=config.num_kv_blocks,
                block_size=config.block_size,
            )
            self.scheduler = BatchScheduler(
                SchedulerConfig(max_num_seqs=config.max_batch_size),
                self.kv_manager,
            )
        else:
            self.scheduler = None
            self.kv_manager = None

    def _load_trellis_pipeline(self, config: EngineConfig):
        """Load a Trellis-quantized model.

        Returns a pipeline wrapper with the same interface as MarlinPipeline.
        """
        # TODO: Import and use TrellisGenerator when ready
        # from ..trellis_generate import TrellisGenerator, GenerationConfig
        # from ..trellis_model import TrellisModel
        #
        # model = TrellisModel.from_pretrained(config.model_path)
        # return TrellisGenerator(model, tokenizer)

        raise NotImplementedError(
            f"Trellis format detected for {config.model_path}, "
            "but TrellisGenerator is not yet integrated. "
            "Use a Marlin-format model or set METAL_MARLIN_MOCK_MODEL=1"
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        """Handle /v1/chat/completions endpoint."""
        # Apply chat template
        prompt = self.pipeline.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        if request.stream:
            return self._stream_generate(prompt, request)

        # Use batched generation when scheduler is available.
        # Skip batched path for mock pipelines since they don't integrate
        # with the scheduler's inference loop.
        if self.scheduler is not None and not isinstance(self.pipeline, _MockPipeline):
            return await self._batched_generate(prompt, request)

        return await self._generate(prompt, request)

    async def completion(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse | AsyncIterator[CompletionResponse]:
        """Handle /v1/completions endpoint."""
        if request.stream:
            return self._stream_completion(request)
        return await self._completion(request)

    async def _generate(
        self,
        prompt: str,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        gen_request = self._track_request(
            prompt,
            request.max_tokens or 256,
            request.temperature,
            request.top_p,
            request.stop or [],
        )
        gen_request.status = RequestStatus.RUNNING
        try:
            result = await asyncio.wait_for(
                self._run_pipeline(
                    prompt,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
                timeout=self.config.request_timeout,
            )
        except TimeoutError:
            gen_request.status = RequestStatus.FINISHED
            raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")
        gen_request.status = RequestStatus.FINISHED

        completion_text, stopped = self._apply_stop_sequences(
            self._strip_prompt(result, prompt),
            request.stop or [],
        )

        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(completion_text)

        finish_reason = "stop" if stopped else "length"
        if completion_tokens < (request.max_tokens or 256):
            finish_reason = "stop"

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=self.model_name,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": finish_reason,
                }
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _completion(self, request: CompletionRequest) -> CompletionResponse:
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

        results: list[str] = []
        for prompt in prompts:
            gen_request = self._track_request(
                prompt,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.stop or [],
            )
            gen_request.status = RequestStatus.RUNNING
            full_text = await self._run_pipeline(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            gen_request.status = RequestStatus.FINISHED
            completion_text, _ = self._apply_stop_sequences(
                self._strip_prompt(full_text, prompt),
                request.stop or [],
            )
            results.append(completion_text)

        prompt_tokens = sum(self._count_tokens(p) for p in prompts)
        completion_tokens = sum(self._count_tokens(r) for r in results)

        choices = [
            {
                "index": idx,
                "text": text,
                "finish_reason": "stop",
            }
            for idx, text in enumerate(results)
        ]

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=self.model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _stream_generate(
        self,
        prompt: str,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        gen_request = self._track_request(
            prompt,
            request.max_tokens or 256,
            request.temperature,
            request.top_p,
            request.stop or [],
        )
        gen_request.status = RequestStatus.RUNNING
        start_time = time.time()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _run_stream() -> None:
            try:
                iterator = self.pipeline(
                    prompt,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True,
                )
                for piece in iterator:
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(self._executor, _run_stream)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        yield ChatCompletionChunk(
            id=request_id,
            created=created,
            model=self.model_name,
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        )

        accumulated = ""
        emitted_len = 0
        stop_sequences = request.stop or []

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.request_timeout:
                gen_request.status = RequestStatus.FINISHED
                raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")
            piece = await queue.get()
            if piece is None:
                break

            accumulated += piece
            emit_text, stopped = self._apply_stop_sequences(accumulated, stop_sequences)
            if stopped:
                new_text = emit_text[emitted_len:]
                if new_text:
                    yield ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=self.model_name,
                        choices=[
                            {"index": 0, "delta": {"content": new_text}, "finish_reason": None}
                        ],
                    )
                break

            yield ChatCompletionChunk(
                id=request_id,
                created=created,
                model=self.model_name,
                choices=[{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            )
            emitted_len += len(piece)

        yield ChatCompletionChunk(
            id=request_id,
            created=created,
            model=self.model_name,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        gen_request.status = RequestStatus.FINISHED

    async def _stream_completion(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionResponse]:
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        start_time = time.time()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _run_stream() -> None:
            try:
                iterator = self.pipeline(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True,
                )
                for piece in iterator:
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(self._executor, _run_stream)

        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.request_timeout:
                raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")
            piece = await queue.get()
            if piece is None:
                break

            yield CompletionResponse(
                id=request_id,
                created=created,
                model=self.model_name,
                choices=[{"index": 0, "text": piece, "finish_reason": None}],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        yield CompletionResponse(
            id=request_id,
            created=created,
            model=self.model_name,
            choices=[{"index": 0, "text": "", "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def _run_pipeline(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.pipeline(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )

    def _strip_prompt(self, full_text: str, prompt: str) -> str:
        if full_text.startswith(prompt):
            return full_text[len(prompt) :]
        return full_text

    def _apply_stop_sequences(self, text: str, stops: list[str]) -> tuple[str, bool]:
        if not stops:
            return text, False

        stop_pos = None
        for stop in stops:
            if not stop:
                continue
            idx = text.find(stop)
            if idx == -1:
                continue
            if stop_pos is None or idx < stop_pos:
                stop_pos = idx

        if stop_pos is None:
            return text, False

        return text[:stop_pos], True

    def _count_tokens(self, text: str) -> int:
        return len(self.pipeline.tokenizer.encode(text))

    def _track_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str],
    ) -> GenerationRequest:
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        prompt_tokens = self.pipeline.tokenizer.encode(prompt)
        req = GenerationRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )
        req.status = RequestStatus.PENDING
        return req

    def get_model_info(self) -> dict:
        """Return detailed model information.

        Used by /v1/models/{model_id} endpoint.
        """
        vocab_size = 32000
        hidden_size = 4096
        num_layers = 32

        if hasattr(self.pipeline, "config"):
            config = self.pipeline.config
            vocab_size = getattr(config, "vocab_size", vocab_size)
            hidden_size = getattr(config, "hidden_size", hidden_size)
            num_layers = getattr(config, "num_hidden_layers", num_layers)

        quant_type = "fp4" if self._model_format == "trellis" else "int4"

        memory_mb = 0
        try:
            import torch

            if torch.backends.mps.is_available():
                memory_mb = torch.mps.current_allocated_memory() // (1024 * 1024)
        except Exception:
            pass

        return {
            "id": self.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "metal-marlin",
            "capabilities": {
                "chat_completions": True,
                "completions": True,
                "streaming": True,
            },
            "config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "quant_type": quant_type,
                "max_context_length": self.config.max_model_len,
                "format": self._model_format,
            },
            "memory_usage_mb": memory_mb,
        }

    async def _batched_generate(
        self,
        prompt: str,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate using continuous batching scheduler.

        This method integrates with the BatchScheduler for efficient
        request processing when batching is enabled.
        """
        from .continuous_batch import RequestPriority

        gen_request = self._track_request(
            prompt,
            request.max_tokens or 256,
            request.temperature,
            request.top_p,
            request.stop or [],
        )

        # Add request to scheduler
        assert self.scheduler is not None
        self.scheduler.add_request(gen_request, priority=RequestPriority.NORMAL)

        # Wait for completion via scheduler
        start_time = time.time()
        while gen_request.status != RequestStatus.FINISHED:
            elapsed = time.time() - start_time
            if elapsed > self.config.request_timeout:
                self.scheduler.abort_request(gen_request.request_id)
                raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")
            await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting

        # Decode output tokens
        completion_tokens = gen_request.output_tokens
        completion_text = self.pipeline.tokenizer.decode(completion_tokens)

        completion_text, stopped = self._apply_stop_sequences(
            completion_text,
            request.stop or [],
        )

        prompt_tokens = self._count_tokens(prompt)
        completion_token_count = len(completion_tokens)

        finish_reason = "stop" if stopped else "length"
        if completion_token_count < (request.max_tokens or 256):
            finish_reason = "stop"

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=self.model_name,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": finish_reason,
                }
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_token_count,
                total_tokens=prompt_tokens + completion_token_count,
            ),
        )
