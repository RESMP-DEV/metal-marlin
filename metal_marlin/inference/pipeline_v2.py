"""Transformers-based inference pipeline with Metal Marlin layer swaps."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._compat import require_torch, torch
from ..quantize_model import quantize_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Transformers is required for this pipeline. Install with: pip install transformers"
        ) from exc


def _is_local_path(model_id_or_path: str) -> bool:
    return Path(model_id_or_path).exists()


def _load_safetensors_keys(path: Path) -> set[str]:
    from safetensors import safe_open

    st_path = path / "model.safetensors"
    if not st_path.exists():
        return set()
    with safe_open(str(st_path), framework="pt") as f:
        return set(f.keys())


def _has_weight_packed_keys(path: Path) -> bool:
    keys = _load_safetensors_keys(path)
    return any(key.endswith(".weight_packed") for key in keys)


def _has_quantization_config(path: Path) -> bool:
    return (path / "quantization_config.json").exists()


def _default_dtype_for_device(device: str) -> Any:
    if torch is None:
        return None
    if device in {"mps", "cuda"}:
        return torch.float16
    return None


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


class TransformersMarlinPipeline:
    """
    Inference pipeline using HuggingFace Transformers + Metal Marlin kernels.

    Unlike the legacy MarlinPipeline which reimplements model structure,
    this uses Transformers for generation and only swaps Linear layers.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "mps",
    ) -> None:
        require_torch()
        assert torch is not None

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        quantize: bool = True,
        bits: int = 4,
        device: str = "mps",
        **quantize_kwargs: Any,
    ) -> TransformersMarlinPipeline:
        """
        Load model from HuggingFace or local path.

        If quantize=True and loading from HF, quantizes on the fly.
        If loading from local quantized checkpoint, loads directly.
        """
        require_torch()
        _require_transformers()
        assert torch is not None

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if bits != 4:
            raise ValueError(f"Only 4-bit Marlin quantization is supported, got bits={bits}")

        dtype = _default_dtype_for_device(device)
        is_local = _is_local_path(model_id_or_path)
        model_path = Path(model_id_or_path) if is_local else None

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)

        if is_local and model_path is not None:
            if _has_weight_packed_keys(model_path):
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                model = quantize_model(
                    model,
                    group_size=quantize_kwargs.get("group_size", 128),
                    skip_layers=quantize_kwargs.get("skip_layers"),
                    layer_filter=quantize_kwargs.get("layer_filter"),
                )
                state_dict = _load_state_dict(model_path)
                model.load_state_dict(state_dict, strict=False)
            elif _has_quantization_config(model_path):
                model = _load_marlin_quantized_model(
                    model_path,
                    device=device,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                if quantize:
                    model = quantize_model(
                        model,
                        group_size=quantize_kwargs.get("group_size", 128),
                        skip_layers=quantize_kwargs.get("skip_layers"),
                        layer_filter=quantize_kwargs.get("layer_filter"),
                    )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            if quantize:
                model = quantize_model(
                    model,
                    group_size=quantize_kwargs.get("group_size", 128),
                    skip_layers=quantize_kwargs.get("skip_layers"),
                    layer_filter=quantize_kwargs.get("layer_filter"),
                )

        return cls(model=model, tokenizer=tokenizer, device=device)

    def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stream: bool = False,
        **generate_kwargs: Any,
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

        generate_kwargs = dict(generate_kwargs)
        generate_kwargs.setdefault("max_new_tokens", max_tokens)
        generate_kwargs.setdefault("temperature", temperature)
        generate_kwargs.setdefault("top_p", top_p)
        if top_k:
            generate_kwargs.setdefault("top_k", top_k)
        if "do_sample" not in generate_kwargs:
            generate_kwargs["do_sample"] = temperature > 0 and (top_p < 1.0 or top_k > 0)
        if (
            "pad_token_id" not in generate_kwargs
            and getattr(self.tokenizer, "pad_token_id", None) is None
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        if stream:
            from threading import Thread

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

    def chat(
        self,
        messages: list[dict[str, Any]],
        **generate_kwargs: Any,
    ) -> str | Iterator[str]:
        """Generate text for chat-formatted models."""
        prompt = _apply_chat_template(self.tokenizer, messages)
        return self(prompt, **generate_kwargs)

    def batch_generate(
        self,
        prompts: list[str],
        **generate_kwargs: Any,
    ) -> list[str]:
        """Batch inference for multiple prompts."""
        require_torch()
        _require_transformers()
        assert torch is not None

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if (
            "pad_token_id" not in generate_kwargs
            and getattr(self.tokenizer, "pad_token_id", None) is None
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def generate(
        self,
        input_ids: torch.Tensor,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Direct access to model.generate()."""
        return self.model.generate(input_ids, **generate_kwargs)


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state_path = model_path / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")
    return load_file(str(state_path))


def _load_marlin_quantized_model(
    model_path: Path,
    *,
    device: str,
) -> PreTrainedModel:
    _require_transformers()
    require_torch()
    assert torch is not None

    from transformers import AutoConfig, AutoModelForCausalLM

    from ..layers import MarlinLinear
    from ..quantize_fp4 import unpack_fp4

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    tensors = _load_state_dict(model_path)
    quant_config = _load_quantization_config(model_path)
    default_group_size = int(quant_config.get("group_size", 128))

    quantized: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for name, tensor in tensors.items():
        if name.endswith(".scales") or name.endswith(".group_size"):
            continue
        scales_name = f"{name}.scales"
        if scales_name in tensors:
            group_size_name = f"{name}.group_size"
            if group_size_name in tensors:
                group_size = int(tensors[group_size_name].view(-1)[0].item())
            else:
                group_size = default_group_size
            quantized[name] = (tensor, tensors[scales_name], group_size)

    for weight_name, (packed, scales, group_size) in quantized.items():
        if not weight_name.endswith(".weight"):
            continue
        module_name = weight_name[: -len(".weight")]
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            module = _get_submodule_fallback(model, module_name)

        if module is None:
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue

        packed_np = packed.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        dequant = unpack_fp4(packed_np, scales_np, group_size=group_size)

        if dequant.shape == (module.in_features, module.out_features):
            weight_fp16 = torch.from_numpy(dequant).T
        else:
            weight_fp16 = torch.from_numpy(dequant)

        weight_fp16 = weight_fp16.to(dtype=torch.float16)
        try:
            packed_new, scales_new = MarlinLinear._pack_fp4_weights(weight_fp16, group_size)
        except ValueError:
            if hasattr(module, "weight") and module.weight.shape == weight_fp16.shape:
                module.weight.data.copy_(weight_fp16)
            continue

        bias = getattr(module, "bias", None)
        quant_layer = MarlinLinear(packed_new, scales_new, bias, group_size=group_size)
        _replace_module(model, module_name, quant_layer)

    float_state = {
        name: tensor
        for name, tensor in tensors.items()
        if not name.endswith(".scales")
        and not name.endswith(".group_size")
        and name not in quantized
    }
    model.load_state_dict(float_state, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_quantization_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "quantization_config.json"
    if not config_path.exists():
        return {}
    import json

    return json.loads(config_path.read_text())


def _replace_module(model: Any, module_name: str, new_module: Any) -> None:
    if not module_name:
        return
    if "." not in module_name:
        setattr(model, module_name, new_module)
        return
    parent_name, child_name = module_name.rsplit(".", 1)
    try:
        parent = model.get_submodule(parent_name)
    except AttributeError:
        parent = _get_submodule_fallback(model, parent_name)
    if parent is None:
        return
    setattr(parent, child_name, new_module)


def _get_submodule_fallback(model: Any, name: str) -> Any | None:
    modules = dict(model.named_modules())
    return modules.get(name)
