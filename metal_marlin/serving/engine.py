from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..inference.mmfp4_pipeline import MMFP4Pipeline
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

# Import C++ serving extension if available
try:
    from ..serving_cpp import ServingCppDispatcher, ServingCppEngineWrapper
    HAS_SERVING_CPP = True
except ImportError:
    HAS_SERVING_CPP = False
    ServingCppDispatcher = None  # type: ignore
    ServingCppEngineWrapper = None  # type: ignore


@dataclass
class RequestLatencyMetrics:
    """Per-request latency tracking."""

    request_id: str
    arrival_time: float = field(default_factory=time.time)
    prefill_start: float | None = None
    prefill_end: float | None = None
    first_token_time: float | None = None
    completion_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def time_to_first_token(self) -> float | None:
        """Time from arrival to first token (TTFT)."""
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def prefill_latency(self) -> float | None:
        """Time spent in prefill phase."""
        if self.prefill_start and self.prefill_end:
            return self.prefill_end - self.prefill_start
        return None

    @property
    def decode_latency(self) -> float | None:
        """Time spent generating tokens after prefill."""
        if self.first_token_time and self.completion_time:
            return self.completion_time - self.first_token_time
        return None

    @property
    def total_latency(self) -> float | None:
        """Total time from arrival to completion."""
        if self.completion_time:
            return self.completion_time - self.arrival_time
        return None

    @property
    def tokens_per_second(self) -> float | None:
        """Average token generation rate."""
        if self.completion_tokens > 0 and self.decode_latency:
            return self.completion_tokens / self.decode_latency
        return None


# Model family detection constants
QWEN_DELTANET_MODEL_TYPES = frozenset({
    "qwen3_5_moe",
    "qwen3_6_moe",
    "qwen3_next",
    "qwen3_vl_moe",
})

QWEN_DELTANET_MODEL_NAME_PATTERNS = frozenset({
    "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b",
    "qwen3.5",
    "qwen3.6",
})

# MoE marker keys for config-driven detection
MOE_EXPERT_COUNTS = frozenset({
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
})

# MoE intermediate size keys
MOE_INTERMEDIATE_KEYS = frozenset({
    "moe_intermediate_size",
    "expert_intermediate_size",
})

# DeltaNet hybrid layer type markers
DELTANET_LAYER_TYPES = frozenset({
    "linear_attention",
    "full_attention",
    "delta_attention",
    "hybrid_attention",
})


def _get_config_value(config: dict, *keys: str, default: Any = None) -> Any:
    """Get a value from config dict, checking multiple possible keys.

    Handles nested text_config dictionaries for multimodal models.
    """
    # First check text_config if present (for multimodal models)
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        for key in keys:
            if key in text_cfg:
                return text_cfg[key]

    # Then check top-level
    for key in keys:
        if key in config:
            return config[key]

    return default


def _is_moe_config(config: dict) -> bool:
    """Check if config indicates a Mixture of Experts model."""
    for key in MOE_EXPERT_COUNTS:
        val = _get_config_value(config, key, default=0)
        try:
            if int(val) > 1:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _has_deltanet_layers(config: dict) -> bool:
    """Check if config has DeltaNet hybrid layer types."""
    layer_types = _get_config_value(config, "layer_types")
    if isinstance(layer_types, list) and layer_types:
        # Check if any layer type matches DeltaNet patterns
        types_set = frozenset(str(lt).lower() for lt in layer_types)
        return bool(types_set & DELTANET_LAYER_TYPES)

    # Also check for full_attention_interval as a secondary marker
    # Qwen3.5 uses int (e.g. 7), Qwen3.6 uses list (e.g. [0, 1, 2, 3])
    fai = _get_config_value(config, "full_attention_interval")
    if fai is not None:
        try:
            if isinstance(fai, list):
                if len(fai) > 0:
                    return True
            elif int(fai) > 0:
                return True
        except (TypeError, ValueError):
            pass

    # Check for explicit use_delta marker (Qwen3.6 DeltaNet flag)
    use_delta = _get_config_value(config, "use_delta")
    if use_delta is True:
        return True

    return False


def _is_qwen_deltanet_family(config: dict, model_name: str = "") -> bool:
    """Check if config represents a Qwen DeltaNet family model.

    Uses config markers instead of hard-coded vocab size heuristics.
    """
    model_type = _get_config_value(config, "model_type", default="").lower()
    model_name_lower = model_name.lower()

    # 1. Check model_type patterns for Qwen DeltaNet family
    for qtype in QWEN_DELTANET_MODEL_TYPES:
        if qtype in model_type:
            return True

    # 2. Check model name patterns
    for pattern in QWEN_DELTANET_MODEL_NAME_PATTERNS:
        if pattern in model_name_lower:
            return True

    # 3. Check architecture list for Qwen DeltaNet markers
    architectures = _get_config_value(config, "architectures", default=[])
    if isinstance(architectures, list):
        arch_blob = " ".join(str(a).lower() for a in architectures)
        for qtype in QWEN_DELTANET_MODEL_TYPES:
            if qtype.replace("_", "") in arch_blob.replace("_", ""):
                return True

    # 4. Config-driven hybrid MoE + DeltaNet detection
    # Qwen3.5/3.6 hybrids have MoE + DeltaNet layer markers
    has_moe = _is_moe_config(config)
    has_deltanet = _has_deltanet_layers(config)
    has_moe_intermediate = any(
        _get_config_value(config, key) is not None
        for key in MOE_INTERMEDIATE_KEYS
    )

    # If config has both MoE markers and DeltaNet layer types, it's a Qwen hybrid
    if has_moe and has_deltanet:
        return True

    # 5. Shared expert intermediate size is distinctive for Qwen hybrid MoE
    shared_exp_size = _get_config_value(
        config,
        "shared_expert_intermediate_size",
        "shared_expert_ffn_hidden_size",
    )
    if shared_exp_size is not None and has_deltanet:
        return True

    return False


def _detect_model_format(model_path: str) -> str:
    """Detect if model is Marlin, MMFP4, or Trellis format.

    Uses config-driven detection based on model_type, text_config,
    layer_types, and MoE metadata instead of narrow hard-coded heuristics.

    Returns:
        'mmfp4' for GLM-4.7-Flash and Qwen DeltaNet hybrid MMFP4 checkpoints
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
            if not isinstance(config, dict):
                config = {}

            quantization_config = config.get("quantization_config", {})
            if not isinstance(quantization_config, dict):
                quantization_config = {}

            # MMFP4 quantization marker detection
            quant_markers = (
                quantization_config.get("quant_method"),
                quantization_config.get("format"),
                quantization_config.get("type"),
                config.get("quant_method"),
                config.get("quant_type"),
                config.get("format"),
            )
            has_mmfp4_quant = any(
                marker is not None and "mmfp4" in str(marker).lower()
                for marker in quant_markers
            )

            model_name = str(config.get("_name_or_path", "")).lower()

            # GLM-4.7-Flash MMFP4 detection using config markers:
            # - GLM architecture markers (glm4_moe or glm4 + moe in model_type/architectures)
            # - MLA attention markers (kv_lora_rank)
            # - MoE structural markers
            architectures = config.get("architectures", [])
            if not isinstance(architectures, list):
                architectures = [architectures]
            architecture_blob = " ".join(str(item).lower() for item in architectures)
            model_type = str(config.get("model_type", "")).lower()

            has_glm47_architecture = (
                "glm-4.7-flash" in model_name
                or "glm4_moe" in model_type
                or ("glm4" in model_type and "moe" in model_type)
                or ("glm4" in architecture_blob and "moe" in architecture_blob)
            )

            # MLA attention marker for GLM models
            has_mla = _get_config_value(config, "kv_lora_rank") is not None

            # Qwen DeltaNet family MMFP4 detection using config markers:
            # - model_type contains Qwen DeltaNet family types
            # - model name contains Qwen3.5/3.6 patterns
            # - layer_types indicates DeltaNet hybrid layers
            # - MoE metadata present alongside DeltaNet markers
            has_qwen_deltanet = _is_qwen_deltanet_family(config, model_name)

            # If MMFP4 quantization AND any recognized MMFP4 architecture:
            # - GLM-4.7-Flash (MLA + MoE)
            # - Qwen DeltaNet family (hybrid MoE + DeltaNet layers)
            # - Generic MoE + MLA combination
            if has_mmfp4_quant:
                expert_count_raw = _get_config_value(
                    config, "num_experts", "num_local_experts", "n_routed_experts", default=0,
                )
                try:
                    expert_count = int(expert_count_raw)
                except (TypeError, ValueError):
                    expert_count = 0
                has_moe = expert_count > 1

                if has_glm47_architecture or has_qwen_deltanet or (has_mla and has_moe):
                    return "mmfp4"

            # Trellis models have specific markers
            if quantization_config.get("quant_method") == "trellis":
                return "trellis"
            if config.get("format") == "trellis":
                return "trellis"
        except (json.JSONDecodeError, KeyError):
            pass

    # Check for Trellis directory structure (sharded layers)
    if (path / "layer_0000").exists():
        return "trellis"

    return "marlin"


def _normalize_model_name(model_path: str) -> str:
    """Normalize model name for API compatibility.

    Maps Qwen DeltaNet family model identifiers to canonical API names
    without inventing new incompatible naming schemes.
    """
    model_name = model_path.split("/")[-1] if model_path else "mock-model"
    model_name_lower = model_name.lower()

    # --- Qwen3.5 variants (check specific patterns before generic) ---
    if "qwen3.5" in model_name_lower or "qwen3_5" in model_name_lower:
        # Specific size variants first
        if "35b" in model_name_lower and "a3b" in model_name_lower:
            return "Qwen/Qwen3.5-35B-A3B"
        if "30b" in model_name_lower and "a3b" in model_name_lower:
            return "Qwen/Qwen3.5-30B-A3B"
        # Generic Qwen3.5 for other variants
        return "Qwen/Qwen3.5"

    # --- Qwen3.6 variants (check specific patterns before generic) ---
    if "qwen3.6" in model_name_lower or "qwen3_6" in model_name_lower:
        # Specific size variants first
        if "35b" in model_name_lower and "a3b" in model_name_lower:
            return "Qwen/Qwen3.6-35B-A3B"
        # Generic Qwen3.6 for other variants
        return "Qwen/Qwen3.6"

    return model_name


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
    use_paged_attention: bool = True
    use_cpp_serving: bool = True  # Use C++ serving path when available


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

        # Normalize the selected model name without discarding explicit overrides.
        model_name = _normalize_model_name(model_name)

        model_format = _detect_model_format(config.model_path)
        self._model_format = model_format

        if use_mock:
            self.pipeline = _MockPipeline(model_name)
        elif model_format == "mmfp4":
            self.pipeline = self._load_mmfp4_pipeline(config)
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
        self._active_requests: dict[str, RequestLatencyMetrics] = {}
        self._latency_history: deque[RequestLatencyMetrics] = deque(maxlen=1000)

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

        # Initialize C++ serving mode if enabled and available
        self._serving_cpp: ServingCppDispatcher | None = None
        self._serving_cpp_wrapper: ServingCppEngineWrapper | None = None
        if config.use_cpp_serving and HAS_SERVING_CPP:
            try:
                self._serving_cpp = ServingCppDispatcher()
                if self._serving_cpp.available:
                    self._serving_cpp_wrapper = ServingCppEngineWrapper(self)
            except Exception:
                pass  # Fall back to Python path

    @property
    def has_cpp_serving(self) -> bool:
        """Return True if C++ serving path is active."""
        return self._serving_cpp is not None and self._serving_cpp.available

    def get_serving_cpp_metrics(self) -> dict[str, Any]:
        """Get C++ serving metrics if available."""
        if self._serving_cpp is not None and self._serving_cpp.available:
            return self._serving_cpp.get_metrics()
        return {"dispatch_count": 0, "total_dispatch_us": 0, "avg_dispatch_us": 0}

    def _load_mmfp4_pipeline(self, config: EngineConfig) -> MMFP4Pipeline:
        """Load an MMFP4 pipeline for GLM-4.7-Flash style checkpoints."""
        if not Path(config.model_path).exists():
            raise FileNotFoundError(f"Model not found: {config.model_path}")

        try:
            pipeline = MMFP4Pipeline.from_pretrained(
                config.model_path,
                device=config.device,
                use_paged_attention=config.use_paged_attention,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MMFP4 model from {config.model_path}: {e}"
            ) from e

        return pipeline

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
        prompt_val = self.pipeline.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = str(prompt_val)

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
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        metrics = RequestLatencyMetrics(request_id=request_id)
        self._active_requests[request_id] = metrics

        gen_request = self._track_request(
            prompt,
            request.max_tokens or 256,
            request.temperature,
            request.top_p,
            request.stop or [],
        )
        gen_request.status = RequestStatus.RUNNING

        metrics.prefill_start = time.time()
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
            metrics.prefill_end = time.time()
            metrics.first_token_time = metrics.prefill_end
        except TimeoutError:
            gen_request.status = RequestStatus.FINISHED
            metrics.completion_time = time.time()
            self._latency_history.append(metrics)
            self._active_requests.pop(request_id, None)
            raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")
        gen_request.status = RequestStatus.FINISHED

        completion_text, stopped = self._apply_stop_sequences(
            self._strip_prompt(result, prompt),
            request.stop or [],
        )

        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(completion_text)

        metrics.prompt_tokens = prompt_tokens
        metrics.completion_tokens = completion_tokens
        metrics.completion_time = time.time()
        self._latency_history.append(metrics)
        self._active_requests.pop(request_id, None)

        finish_reason = "stop" if stopped else "length"
        if completion_tokens < (request.max_tokens or 256):
            finish_reason = "stop"

        return ChatCompletionResponse(
            id=request_id,
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
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        metrics = RequestLatencyMetrics(request_id=request_id)
        self._active_requests[request_id] = metrics

        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

        metrics.prefill_start = time.time()
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

        metrics.prefill_end = time.time()
        metrics.first_token_time = metrics.prefill_end

        prompt_tokens = sum(self._count_tokens(p) for p in prompts)
        completion_tokens = sum(self._count_tokens(r) for r in results)

        metrics.prompt_tokens = prompt_tokens
        metrics.completion_tokens = completion_tokens
        metrics.completion_time = time.time()
        self._latency_history.append(metrics)
        self._active_requests.pop(request_id, None)

        choices = [
            {
                "index": idx,
                "text": text,
                "finish_reason": "stop",
            }
            for idx, text in enumerate(results)
        ]

        return CompletionResponse(
            id=request_id,
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
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        metrics = RequestLatencyMetrics(request_id=request_id)
        self._active_requests[request_id] = metrics

        gen_request = self._track_request(
            prompt,
            request.max_tokens or 256,
            request.temperature,
            request.top_p,
            request.stop or [],
        )
        gen_request.status = RequestStatus.RUNNING

        metrics.prefill_start = time.time()
        metrics.prompt_tokens = self._count_tokens(prompt)

        start_time = time.time()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _run_stream() -> None:
            try:
                stream_output = self._call_pipeline(
                    prompt,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True,
                )
                iterator = (
                    iter([stream_output]) if isinstance(stream_output, str) else stream_output
                )
                for piece in iterator:
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(self._executor, _run_stream)

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
        first_chunk = True

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.request_timeout:
                gen_request.status = RequestStatus.FINISHED
                metrics.completion_time = time.time()
                self._latency_history.append(metrics)
                self._active_requests.pop(request_id, None)
                raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")
            piece = await queue.get()
            if piece is None:
                break

            if first_chunk:
                metrics.prefill_end = time.time()
                metrics.first_token_time = metrics.prefill_end
                first_chunk = False

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

        metrics.completion_tokens = self._count_tokens(accumulated)
        metrics.completion_time = time.time()
        self._latency_history.append(metrics)
        self._active_requests.pop(request_id, None)

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
                stream_output = self._call_pipeline(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True,
                )
                iterator = (
                    iter([stream_output]) if isinstance(stream_output, str) else stream_output
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
        result = await loop.run_in_executor(
            self._executor,
            lambda: self._call_pipeline(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        return str(result)

    def _call_pipeline(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool = False,
    ) -> str | Iterator[str]:
        if self._model_format == "mmfp4":
            return self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
        return self.pipeline(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
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

        quant_type = "fp4" if self._model_format in {"trellis", "mmfp4"} else "int4"

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

    def get_latency_stats(self) -> dict:
        """Get aggregated latency statistics across all requests.

        Returns:
            Dictionary with p50, p95, p99 latencies and throughput metrics.
        """
        if not self._latency_history:
            return {
                "total_requests": 0,
                "avg_ttft_ms": 0.0,
                "avg_prefill_ms": 0.0,
                "avg_decode_ms": 0.0,
                "avg_total_ms": 0.0,
                "avg_tokens_per_second": 0.0,
            }

        ttfts = [m.time_to_first_token for m in self._latency_history if m.time_to_first_token]
        prefills = [m.prefill_latency for m in self._latency_history if m.prefill_latency]
        decodes = [m.decode_latency for m in self._latency_history if m.decode_latency]
        totals = [m.total_latency for m in self._latency_history if m.total_latency]
        tps = [m.tokens_per_second for m in self._latency_history if m.tokens_per_second]

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f < len(sorted_data) - 1 else f
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

        return {
            "total_requests": len(self._latency_history),
            "ttft_p50_ms": percentile(ttfts, 50) * 1000 if ttfts else 0.0,
            "ttft_p95_ms": percentile(ttfts, 95) * 1000 if ttfts else 0.0,
            "ttft_p99_ms": percentile(ttfts, 99) * 1000 if ttfts else 0.0,
            "prefill_p50_ms": percentile(prefills, 50) * 1000 if prefills else 0.0,
            "decode_p50_ms": percentile(decodes, 50) * 1000 if decodes else 0.0,
            "total_p50_ms": percentile(totals, 50) * 1000 if totals else 0.0,
            "total_p95_ms": percentile(totals, 95) * 1000 if totals else 0.0,
            "total_p99_ms": percentile(totals, 99) * 1000 if totals else 0.0,
            "avg_tokens_per_second": sum(tps) / len(tps) if tps else 0.0,
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

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        metrics = RequestLatencyMetrics(request_id=request_id)
        self._active_requests[request_id] = metrics

        gen_request = self._track_request(
            prompt,
            request.max_tokens or 256,
            request.temperature,
            request.top_p,
            request.stop or [],
        )

        metrics.prefill_start = time.time()
        metrics.prompt_tokens = self._count_tokens(prompt)

        # Add request to scheduler
        assert self.scheduler is not None
        self.scheduler.add_request(gen_request, priority=RequestPriority.NORMAL)

        # Wait for completion via scheduler
        start_time = time.time()
        first_token_tracked = False
        while gen_request.status != RequestStatus.FINISHED:
            elapsed = time.time() - start_time
            if elapsed > self.config.request_timeout:
                self.scheduler.abort_request(gen_request.request_id)
                metrics.completion_time = time.time()
                self._latency_history.append(metrics)
                self._active_requests.pop(request_id, None)
                raise RequestTimeoutError(f"Request timed out after {self.config.request_timeout}s")

            if not first_token_tracked and gen_request.output_tokens:
                metrics.prefill_end = time.time()
                metrics.first_token_time = metrics.prefill_end
                first_token_tracked = True

            await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting

        # Decode output tokens
        completion_tokens = gen_request.output_tokens
        completion_text = str(self.pipeline.tokenizer.decode(completion_tokens))

        completion_text, stopped = self._apply_stop_sequences(
            completion_text,
            request.stop or [],
        )

        prompt_tokens = self._count_tokens(prompt)
        completion_token_count = len(completion_tokens)

        metrics.completion_tokens = completion_token_count
        metrics.completion_time = time.time()
        self._latency_history.append(metrics)
        self._active_requests.pop(request_id, None)

        finish_reason = "stop" if stopped else "length"
        if completion_token_count < (request.max_tokens or 256):
            finish_reason = "stop"

        return ChatCompletionResponse(
            id=request_id,
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
