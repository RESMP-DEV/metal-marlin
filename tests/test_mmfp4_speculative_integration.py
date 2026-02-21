from __future__ import annotations

import torch

from metal_marlin.inference.mmfp4_speculative import speculative_decode


class _MockOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _MockModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, target_token: int = 7):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.target_token = target_token

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.hidden_size, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor) -> _MockOutput:
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -20.0, dtype=torch.float32)
        logits[:, :, self.target_token] = 20.0
        return _MockOutput(logits)


class _MockMTPHead(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, draft_tokens: list[int] | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.draft_tokens = draft_tokens or [7, 8, 9, 10]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        num_draft = len(self.draft_tokens)
        logits = torch.full((batch_size, num_draft, self.vocab_size), -20.0, dtype=torch.float32)
        for i, token in enumerate(self.draft_tokens):
            logits[:, i, token] = 20.0
        return logits


def test_speculative_decode_runs_and_returns_valid_tokens_with_positive_acceptance():
    model = _MockModel(vocab_size=32, hidden_size=16, target_token=7)
    mtp_head = _MockMTPHead(vocab_size=32, draft_tokens=[7, 8, 9, 10])
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    new_tokens, num_accepted = speculative_decode(
        model=model,
        mtp_head=mtp_head,
        input_ids=input_ids,
        num_draft=4,
        temperature=0.0,
    )

    assert isinstance(new_tokens, torch.Tensor)
    assert new_tokens.ndim == 2
    assert new_tokens.shape[0] == 1
    assert new_tokens.shape[1] >= 1
    assert new_tokens.dtype == torch.long

    assert int(new_tokens.min()) >= 0
    assert int(new_tokens.max()) < model.vocab_size

    assert isinstance(num_accepted, int)
    assert num_accepted > 0
    acceptance_rate = num_accepted / 4
    assert acceptance_rate > 0.0
