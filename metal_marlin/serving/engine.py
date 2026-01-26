from __future__ import annotations

import asyncio
import concurrent.futures
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..inference.pipeline import MarlinPipeline
from .openai_schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from .request import GenerationRequest, RequestStatus


@dataclass
class EngineConfig:
    model_path: str
    device: str = "mps"
    max_model_len: int = 4096
    max_batch_size: int = 32


class ServingEngine:
    """Async-compatible inference engine for OpenAI API.

    This implementation bridges the MarlinPipeline to the serving layer and
    provides async request handling with optional streaming.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.pipeline = MarlinPipeline.from_pretrained(
            config.model_path, device=config.device
        )
        self.model_name = str(config.model_path).split("/")[-1]
        self._request_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self._results: dict[str, asyncio.Future] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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
        result = await self._run_pipeline(
            prompt,
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature,
            top_p=request.top_p,
        )
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
