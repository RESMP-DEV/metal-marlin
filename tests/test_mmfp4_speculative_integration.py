from __future__ import annotations
import logging

import torch

from metal_marlin.inference.mmfp4_speculative import speculative_decode



logger = logging.getLogger(__name__)

class _MockOutput:
    def __init__(self, logits: torch.Tensor):
        logger.debug("initializing %s with logits=%s", type(self).__name__, logits)
        self.logits = logits


class _MockModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, target_token: int = 7):
        logger.debug("initializing %s with vocab_size=%s, hidden_size=%s, target_token=%s", type(self).__name__, vocab_size, hidden_size, target_token)
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.target_token = target_token

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        logger.debug("get_hidden called with input_ids=%s", input_ids)
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.hidden_size, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor) -> _MockOutput:
        logger.debug("forward: input shape=%s dtype=%s", input_ids.shape if hasattr(input_ids, "shape") else type(input_ids).__name__, input_ids.dtype if hasattr(input_ids, "dtype") else "N/A")
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -20.0, dtype=torch.float32)
        logits[:, :, self.target_token] = 20.0
        return _MockOutput(logits)


class _MockMTPHead(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, draft_tokens: list[int] | None = None):
        logger.debug("initializing %s with vocab_size=%s, draft_tokens=%s", type(self).__name__, vocab_size, draft_tokens)
        super().__init__()
        self.vocab_size = vocab_size
        self.draft_tokens = draft_tokens or [7, 8, 9, 10]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logger.debug("forward: input shape=%s dtype=%s", hidden_states.shape if hasattr(hidden_states, "shape") else type(hidden_states).__name__, hidden_states.dtype if hasattr(hidden_states, "dtype") else "N/A")
        batch_size = hidden_states.shape[0]
        num_draft = len(self.draft_tokens)
        logits = torch.full((batch_size, num_draft, self.vocab_size), -20.0, dtype=torch.float32)
        for i, token in enumerate(self.draft_tokens):
            logits[:, i, token] = 20.0
        return logits


def test_speculative_decode_runs_and_returns_valid_tokens_with_positive_acceptance():
    logger.info("running test_speculative_decode_runs_and_returns_valid_tokens_with_positive_acceptance")
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
