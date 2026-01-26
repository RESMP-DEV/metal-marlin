import torch
from transformers import AutoTokenizer

from .models import QuantizedGLM4MoE, QuantizedLlama
from .models.nemotron import QuantizedNemotron
from .models.qwen3_dense import QuantizedQwen3Dense
from .models.qwen3_moe import QuantizedQwen3MoE
from .quantized_loader import QuantizedModel


class MetalInferenceEngine:
    """High-level inference API for quantized models."""

    MODEL_CLASSES = {
        "glm4_moe_lite": QuantizedGLM4MoE,
        "llama": QuantizedLlama,
        "qwen3": QuantizedQwen3Dense,
        "qwen3_moe": QuantizedQwen3MoE,
        "nemotron_h": QuantizedNemotron,
    }

    def __init__(self, model_path: str, device: str = "mps"):
        self.quantized = QuantizedModel.load(model_path)
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.quantized.config.get("base_model_id")
        )
        self.device = device

    def _load_model(self):
        arch = self.quantized.config.get("architecture")
        model_class = self.MODEL_CLASSES.get(arch)
        if model_class is None:
            raise ValueError(f"Unsupported architecture: {arch}")
        return model_class(self.quantized)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        # Autoregressive generation loop
        for _ in range(max_tokens):
            logits = self.model(input_ids)
            next_token = self._sample(logits[:, -1], temperature, top_p)
            if next_token == self.tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(input_ids[0])

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / max(temperature, 1e-6)

        if top_p >= 1.0:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        if sorted_mask.shape[-1] > 1:
            sorted_mask[..., 0] = False

        filtered_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered_probs = torch.softmax(filtered_logits, dim=-1)
        next_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        return sorted_indices.gather(-1, next_indices.unsqueeze(-1)).squeeze(-1)
