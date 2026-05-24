"""Compatibility exports for older Trellis configuration imports."""

from __future__ import annotations

from .trellis.config import GLM4_TOKENIZER_ID, TrellisModelConfig


class TrellisConfig(TrellisModelConfig):
    """Backward-compatible alias for :class:`TrellisModelConfig`."""

    def __init__(
        self,
        *args: object,
        num_key_value_heads: int | None = None,
        **kwargs: object,
    ) -> None:
        if num_key_value_heads is not None and "num_kv_heads" not in kwargs:
            kwargs["num_kv_heads"] = num_key_value_heads
        super().__init__(*args, **kwargs)

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    @num_key_value_heads.setter
    def num_key_value_heads(self, value: int) -> None:
        self.num_kv_heads = value


__all__ = ["GLM4_TOKENIZER_ID", "TrellisConfig", "TrellisModelConfig"]
