"""Unit tests for GLM MTP head."""

from __future__ import annotations

import torch

from metal_marlin.layers.mtp_head import GLMMTPHead


def test_mtp_head_creation() -> None:
    head = GLMMTPHead(hidden_size=16, vocab_size=64, num_tokens=4)

    assert head.hidden_size == 16
    assert head.vocab_size == 64
    assert head.num_tokens == 4
    assert len(head.heads) == 4

    for proj in head.heads:
        assert proj.in_features == 16
        assert proj.out_features == 64
        assert proj.bias is None


def test_mtp_head_forward_output_count() -> None:
    head = GLMMTPHead(hidden_size=32, vocab_size=128, num_tokens=3)
    hidden = torch.randn(2, 5, 32)

    outputs = head(hidden)

    assert isinstance(outputs, list)
    assert len(outputs) == 3


def test_mtp_head_forward_output_shapes() -> None:
    head = GLMMTPHead(hidden_size=24, vocab_size=100, num_tokens=4)
    hidden = torch.randn(3, 7, 24)

    outputs = head(hidden)

    for logits in outputs:
        assert logits.shape == (3, 7, 100)
