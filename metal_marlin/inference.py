import torch
from transformers import AutoTokenizer

from .models import QuantizedGLM4MoE, QuantizedLlama
from .models.nemotron import QuantizedNemotron
from .models.qwen3_dense import QuantizedQwen3Dense
from .models.qwen3_moe import QuantizedQwen3MoE
from .quantized_loader import QuantizedModel
from .sampler import MetalSampler


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
        self.sampler = MetalSampler(
            vocab_size=self.quantized.config.get("vocab_size", 32000)
        )

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

    def _sample(
        self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int = 0
    ) -> torch.Tensor:
        if temperature == 0:
            token_id = self.sampler.argmax(logits)
        elif top_p < 1.0:
            token_id = self.sampler.sample_top_p(logits, top_p, temperature)
        elif top_k > 0:
            token_id = self.sampler.sample_top_k(logits, top_k, temperature)
        else:
            token_id = self.sampler.sample_categorical(logits, temperature)
        return torch.tensor(token_id, device=logits.device)
