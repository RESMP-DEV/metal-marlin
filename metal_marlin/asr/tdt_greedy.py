"""
TDT Greedy Decoding Implementation.

Transducer-Duration-Transducer (TDT) greedy decoding uses duration predictions
to skip multiple frames at once, unlike traditional RNN-T decoding.
"""

from typing import Any

import torch


def tdt_greedy_decode(
    encoder_out: torch.Tensor,  # [1, T, enc_dim]
    predictor: Any,  # TDTPredictor
    joint: Any,  # TDTJoint
    blank_id: int = 0,
    max_symbols_per_step: int = 10,
) -> list[int]:
    """
    Greedy decode with TDT duration skipping.

    At each (t, u):
    1. Compute joint output
    2. If argmax == blank: t += duration_pred, u stays
    3. Else: emit token, u += 1, t stays

    Args:
        encoder_out: Encoder output tensor [1, T, enc_dim]
        predictor: TDT predictor network
        joint: TDT joint network
        blank_id: ID of the blank token
        max_symbols_per_step: Maximum symbols to emit per time step

    Returns:
        List of decoded token IDs
    """
    tokens = []
    t, u = 0, 0
    T = encoder_out.size(1)
    state = None

    while t < T:
        # Get current encoder frame
        enc_frame = encoder_out[:, t : t + 1, :]  # [1, 1, enc_dim]

        symbols_this_step = 0
        while symbols_this_step < max_symbols_per_step:
            # Get predictor output for current position
            pred_out, state = predictor(enc_frame, state)

            # Compute joint output
            joint_out = joint(enc_frame, pred_out)  # [1, 1, vocab_size]

            # Get the most likely token
            log_probs = torch.log_softmax(joint_out, dim=-1)
            max_id = torch.argmax(log_probs, dim=-1).item()

            if max_id == blank_id:
                # Blank token: use duration prediction to skip frames
                if hasattr(predictor, "predict_duration"):
                    duration = predictor.predict_duration(enc_frame, state)
                    duration = max(1, int(duration.item()))  # At least 1 frame
                else:
                    duration = 1  # Fallback to 1 frame if no duration prediction

                t += duration
                break
            else:
                # Non-blank token: emit it and continue at same time step
                tokens.append(max_id)
                u += 1
                symbols_this_step += 1

                # Check if we should continue at this time step
                if u >= max_symbols_per_step:
                    break

    return tokens


def batch_tdt_greedy_decode(
    encoder_out: torch.Tensor,  # [B, T, enc_dim]
    predictor: Any,
    joint: Any,
    blank_id: int = 0,
    max_symbols_per_step: int = 10,
) -> list[list[int]]:
    """
    Batch version of TDT greedy decoding.

    Args:
        encoder_out: Encoder output tensor [B, T, enc_dim]
        predictor: TDT predictor network
        joint: TDT joint network
        blank_id: ID of the blank token
        max_symbols_per_step: Maximum symbols to emit per time step

    Returns:
        List of decoded token sequences, one per batch element
    """
    batch_size = encoder_out.size(0)
    results = []

    for i in range(batch_size):
        single_encoder = encoder_out[i : i + 1, :, :]  # [1, T, enc_dim]
        tokens = tdt_greedy_decode(single_encoder, predictor, joint, blank_id, max_symbols_per_step)
        results.append(tokens)

    return results


class TDTPredictor:
    """
    Mock TDT Predictor interface for testing and compatibility.
    In practice, this would be replaced with the actual implementation.
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def __call__(self, enc_frame: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        """Forward pass of predictor."""
        batch_size, _, enc_dim = enc_frame.shape

        # Mock implementation - replace with actual predictor logic
        pred_out = torch.randn(batch_size, 1, self.hidden_dim)
        new_state = state if state is not None else torch.zeros(batch_size, self.hidden_dim)

        return pred_out, new_state

    def predict_duration(self, enc_frame: torch.Tensor, state: Any = None) -> torch.Tensor:
        """Predict duration for frame skipping."""
        batch_size, _, _ = enc_frame.shape

        # Mock implementation - predict duration between 1 and 5
        duration_logits = torch.randn(batch_size, 1, 5)  # durations 1-5
        duration_probs = torch.softmax(duration_logits, dim=-1)
        duration_values = torch.arange(1, 6, dtype=torch.float32, device=enc_frame.device)

        # Expected value
        duration = torch.sum(duration_probs * duration_values, dim=-1, keepdim=True)

        return duration.squeeze(-1)  # [batch_size]


class TDTJoint:
    """
    Mock TDT Joint interface for testing and compatibility.
    In practice, this would be replaced with the actual implementation.
    """

    def __init__(self, vocab_size: int, enc_dim: int, pred_dim: int):
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim

    def __call__(self, enc_frame: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        """Joint forward pass combining encoder and predictor outputs."""
        batch_size, _, _ = enc_frame.shape

        # Mock implementation - replace with actual joint logic
        joint_dim = self.enc_dim + self.pred_dim
        combined = torch.cat([enc_frame, pred_out], dim=-1)  # [batch, 1, joint_dim]

        # Mock linear layer to vocab size
        joint_out = torch.randn(batch_size, 1, self.vocab_size)

        return joint_out
