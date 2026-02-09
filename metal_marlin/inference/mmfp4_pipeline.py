"""High-level text generation pipeline for MMFP4 causal language models."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Thread
from typing import TYPE_CHECKING, Any, Protocol, cast

from .._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


class MMFP4ForCausalLM(Protocol):
    """Minimal protocol required by MMFP4Pipeline."""

    def to(self, device: str) -> MMFP4ForCausalLM:
        ...

    def eval(self) -> MMFP4ForCausalLM:
        ...

    def generate(self, input_ids: torch_typing.Tensor, **kwargs: Any) -> torch_typing.Tensor:
        ...


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Transformers is required for MMFP4Pipeline. Install with: pip install transformers"
        ) from exc


def _default_dtype_for_device(device: str) -> Any:
    if torch is None:
        return None
    if device in {"mps", "cuda"}:
        return torch.float16
    return None


def _resolve_device(requested_device: str) -> str:
    require_torch()
    assert torch is not None

    if requested_device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def _infer_model_device(model: Any) -> str:
    if hasattr(model, "device"):
        return str(model.device)
    if hasattr(model, "parameters"):
        try:
            first_param = next(model.parameters())
            return str(first_param.device)
        except StopIteration:
            pass
        except TypeError:
            pass
    return "cpu"


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


class MMFP4Pipeline:
    """High-level API for text generation with MMFP4 models."""

    def __init__(self, model: MMFP4ForCausalLM, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = _infer_model_device(model)

        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "mps") -> MMFP4Pipeline:
        """Load model and tokenizer from path."""
        require_torch()
        _require_transformers()
        assert torch is not None

        from transformers import AutoTokenizer

        from ..models.mmfp4_causal_lm import MMFP4ForCausalLM as MMFP4Model

        resolved_device = _resolve_device(device)

        # Use custom MMFP4 model loader (not AutoModelForCausalLM)
        model = cast(
            MMFP4ForCausalLM,
            MMFP4Model.from_pretrained(model_path, device=resolved_device),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        if (
            getattr(tokenizer, "pad_token_id", None) is None
            and getattr(tokenizer, "eos_token_id", None) is not None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer)

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate text from prompt."""
        require_torch()
        _require_transformers()
        assert torch is not None

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        do_sample = temperature > 0 and (top_p < 1.0 or temperature != 1.0)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(float(temperature), 1e-5)
            generate_kwargs["top_p"] = float(top_p)

        if (
            getattr(self.tokenizer, "pad_token_id", None) is None
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        if stream:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs["streamer"] = streamer

            thread = Thread(
                target=self.model.generate,
                kwargs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    **generate_kwargs,
                },
                daemon=True,
            )
            thread.start()
            return streamer

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """Chat completion with message history."""
        prompt = _apply_chat_template(self.tokenizer, messages)
        result = self(prompt, **kwargs)
        if isinstance(result, str):
            return result
        return "".join(result)


__all__ = ["MMFP4ForCausalLM", "MMFP4Pipeline"]
