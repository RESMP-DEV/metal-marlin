"""
Eagle v3 speculative decoding head.

Eagle (Efficient Autoregressive Language model with Learned Exploration) improves
speculative decoding throughput by:
1. Multi-token prediction: predicts k tokens at once from hidden states
2. Tree-structured drafts: explores multiple continuation paths in parallel
3. Dynamic depth adaptation: adjusts tree depth based on acceptance rate

This implements the draft head architecture from:
  EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
  (Li et al., 2024)

The key insight is that a small MLP head attached to the main model's hidden states
can predict multiple future tokens simultaneously, enabling tree-structured
speculation where the verifier evaluates all branches in a single forward pass.

Usage:
    eagle_head = EagleHead(hidden_size=4096, vocab_size=32000)

    # During generation:
    draft_output = eagle_head(hidden_states)
    tree_tokens, tree_mask = eagle_head.build_draft_tree(hidden_states)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..kv_cache import KVCache
from .draft import DraftModel, DraftOutput


@dataclass
class TreeDraftOutput(DraftOutput):
    """Extended DraftOutput for tree-structured speculation.

    In tree-structured speculation, we explore multiple paths through the
    token space. Each path is a sequence of tokens, and all paths share
    common prefixes up to their branching point.

    Attributes:
        tokens: Flattened tree tokens for verification, shape [batch, num_tree_tokens].
        probs: Probability distributions for each tree position,
            shape [batch, num_tree_tokens, vocab_size].
        tree_mask: Causal attention mask for tree structure,
            shape [batch, num_tree_tokens, num_tree_tokens]. Position (i, j) is True
            if position i can attend to position j.
        tree_indices: Parent indices for each node in the flattened tree,
            shape [batch, num_tree_tokens]. Index -1 for root nodes.
        path_lengths: Number of tokens in each path, shape [batch, num_paths].
    """

    tree_mask: Tensor  # [batch, num_tree_tokens, num_tree_tokens]
    tree_indices: Tensor  # [batch, num_tree_tokens]
    path_lengths: Tensor  # [batch, num_paths]


class EagleHead(DraftModel):
    """Eagle v3 draft head for speculative decoding.

    Uses a small MLP to predict multiple tokens from hidden states,
    enabling tree-structured speculation with higher acceptance.

    The head architecture:
      - Input projection: hidden_size -> head_dim
      - N prediction heads: each head_dim -> vocab_size
      - Each head predicts the token at a specific depth

    Tree construction:
      - Start with top-k tokens from depth-0 prediction
      - For each token, predict depth-1 tokens conditioned on it
      - Continue until max_depth, pruning low-probability branches

    The tree is flattened for efficient batched verification, with a
    tree_mask that encodes the parent-child relationships.

    Args:
        hidden_size: Dimension of the model's hidden states.
        vocab_size: Size of the vocabulary.
        num_draft_tokens: Maximum total tokens in the draft tree.
        tree_width: Maximum children per node (top-k branching factor).
        head_dim: Internal dimension of the draft head (default: hidden_size // 4).
        num_layers: Number of layers in the draft MLP.
        dropout: Dropout probability for training.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_draft_tokens: int = 5,
        tree_width: int = 3,
        head_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        device: torch.device | None = None,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_draft_tokens = num_draft_tokens
        self.tree_width = tree_width
        self.head_dim = head_dim or hidden_size // 4
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")

        # Build the draft head MLP
        layers: list[nn.Module] = []
        in_dim = hidden_size
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, self.head_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = self.head_dim

        self.feature_extractor = nn.Sequential(*layers).to(self.device)

        # Multi-token prediction heads: one per depth level
        # For linear drafting, depth = num_draft_tokens
        # For tree drafting, use _compute_max_depth for the tree budget
        # We create heads for linear depth (num_draft_tokens), which also covers tree depth
        self.max_depth = num_draft_tokens

        self.token_heads = nn.ModuleList([
            nn.Linear(self.head_dim if num_layers > 1 else hidden_size, vocab_size)
            for _ in range(num_draft_tokens)
        ]).to(self.device)

        # Embedding layer for predicted tokens to condition subsequent predictions
        self.token_embed = nn.Embedding(vocab_size, self.head_dim).to(self.device)

        # Fusion layer to combine hidden features with token embeddings
        self.fusion = nn.Linear(
            self.head_dim + self.head_dim, self.head_dim
        ).to(self.device)

        # Acceptance rate tracking for adaptive depth
        self._acceptance_history: list[float] = []
        self._adaptive_depth: int = self.max_depth

    @staticmethod
    def _compute_max_depth(num_tokens: int, width: int) -> int:
        """Compute maximum tree depth from total token budget.

        A tree with width w and depth d has at most:
          sum_{i=1}^{d} w^i = w * (w^d - 1) / (w - 1) tokens

        We find the largest d such that this sum <= num_tokens.
        """
        if width == 1:
            return num_tokens

        total = 0
        depth = 0
        level_size = width
        while total + level_size <= num_tokens:
            total += level_size
            depth += 1
            level_size *= width

        return max(1, depth)

    def forward(
        self,
        hidden_states: Tensor,
        tree_mask: Tensor | None = None,
    ) -> DraftOutput:
        """Generate linear draft tokens (non-tree) from hidden states.

        This is the simple interface that matches the DraftModel protocol.
        For tree-structured drafting, use build_draft_tree() instead.

        Args:
            hidden_states: Final hidden states from the target model,
                shape [batch, seq_len, hidden_size]. Typically just [batch, 1, hidden_size]
                for the last position during decode.
            tree_mask: Unused in linear mode; provided for interface compatibility.

        Returns:
            DraftOutput with tokens [batch, num_draft_tokens] and
            probs [batch, num_draft_tokens, vocab_size].
        """
        batch_size = hidden_states.shape[0]

        # Use last position's hidden state
        h = hidden_states[:, -1, :]  # [batch, hidden_size]

        # Extract features
        features = self.feature_extractor(h)  # [batch, head_dim]

        tokens_list: list[Tensor] = []
        probs_list: list[Tensor] = []

        current_features = features

        for depth in range(min(self.num_draft_tokens, len(self.token_heads))):
            # Predict logits for this depth
            logits = self.token_heads[depth](current_features)  # [batch, vocab]
            probs = torch.softmax(logits, dim=-1)

            # Greedy selection
            token = probs.argmax(dim=-1)  # [batch]

            tokens_list.append(token)
            probs_list.append(probs)

            if depth < self.num_draft_tokens - 1:
                # Condition on predicted token for next depth
                token_embed = self.token_embed(token)  # [batch, head_dim]
                fused = self.fusion(torch.cat([current_features, token_embed], dim=-1))
                current_features = fused

        return DraftOutput(
            tokens=torch.stack(tokens_list, dim=1),  # [batch, num_draft]
            probs=torch.stack(probs_list, dim=1),  # [batch, num_draft, vocab]
        )

    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative token proposals.

        This method requires hidden_states rather than input_ids directly.
        In typical usage, the caller computes hidden_states from the target
        model and passes them here. This is a compatibility shim that returns
        uniform proposals when called without hidden state access.

        For proper Eagle usage, call forward() with hidden_states directly.

        Args:
            input_ids: Current context token IDs, shape [batch, seq_len].
            kv_cache: Target model's KV cache (unused in Eagle; we need hidden states).
            num_tokens: Number of tokens to speculatively generate.

        Returns:
            DraftOutput with proposed tokens and their probability distributions.
            Falls back to uniform distribution if hidden states unavailable.
        """
        batch_size = input_ids.shape[0]
        num_tokens = min(num_tokens, self.num_draft_tokens)

        # Without hidden states, return uniform distribution
        # In practice, EagleHead is used via forward() with hidden states
        tokens = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=self.device)
        probs = torch.ones(
            batch_size, num_tokens, self.vocab_size,
            dtype=torch.float32, device=self.device
        ) / self.vocab_size

        return DraftOutput(tokens=tokens, probs=probs)

    def speculate_from_hidden(
        self,
        hidden_states: Tensor,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative proposals from hidden states.

        This is the preferred API for Eagle when hidden states are available.

        Args:
            hidden_states: Hidden states from target model, [batch, seq, hidden_size].
            num_tokens: Number of tokens to generate.

        Returns:
            DraftOutput with tokens and probability distributions.
        """
        original_num = self.num_draft_tokens
        self.num_draft_tokens = min(num_tokens, original_num)
        try:
            result = self.forward(hidden_states)
        finally:
            self.num_draft_tokens = original_num
        return result

    def build_draft_tree(
        self,
        hidden_states: Tensor,
        max_depth: int = 5,
        max_width: int = 3,
        prob_threshold: float = 0.01,
    ) -> tuple[Tensor, Tensor]:
        """Build tree of draft tokens with probability thresholding.

        Constructs a tree-structured draft where each level branches into
        multiple candidate continuations. Low-probability branches are
        pruned to stay within the token budget.

        The tree is returned in a flattened form suitable for batched
        verification, along with a causal mask encoding the tree structure.

        Args:
            hidden_states: Hidden states from target model, [batch, seq, hidden_size].
            max_depth: Maximum depth of the draft tree.
            max_width: Maximum children per node (top-k branching).
            prob_threshold: Minimum probability to keep a branch.

        Returns:
            Tuple of:
              - tree_tokens: Flattened tree tokens, [batch, num_tree_tokens].
              - tree_mask: Causal attention mask, [batch, num_tree_tokens, num_tree_tokens].
        """
        batch_size = hidden_states.shape[0]
        max_depth = min(max_depth, self._adaptive_depth, len(self.token_heads))
        max_width = min(max_width, self.tree_width)

        # Use last position's hidden state
        h = hidden_states[:, -1, :]  # [batch, hidden_size]
        features = self.feature_extractor(h)  # [batch, head_dim]

        # We'll build the tree level by level
        # Each node is represented as (features, parent_idx, accumulated_prob)
        # Level 0: root predictions

        all_tokens: list[list[Tensor]] = [[] for _ in range(batch_size)]
        all_probs: list[list[Tensor]] = [[] for _ in range(batch_size)]
        all_parents: list[list[int]] = [[] for _ in range(batch_size)]

        # Track nodes at current level: list of (features, parent_idx, prob)
        current_level: list[list[tuple[Tensor, int, float]]] = [
            [(features[b], -1, 1.0)] for b in range(batch_size)
        ]

        total_tokens = [0 for _ in range(batch_size)]
        max_tokens = self.num_draft_tokens

        for depth in range(max_depth):
            if depth >= len(self.token_heads):
                break

            head = self.token_heads[depth]
            next_level: list[list[tuple[Tensor, int, float]]] = [[] for _ in range(batch_size)]

            for b in range(batch_size):
                if total_tokens[b] >= max_tokens:
                    continue

                for node_features, parent_idx, parent_prob in current_level[b]:
                    if total_tokens[b] >= max_tokens:
                        break

                    # Predict logits for this node
                    logits = head(node_features.unsqueeze(0)).squeeze(0)  # [vocab]
                    probs = torch.softmax(logits, dim=-1)

                    # Get top-k candidates
                    topk_probs, topk_tokens = probs.topk(max_width)

                    for i in range(max_width):
                        if total_tokens[b] >= max_tokens:
                            break

                        tok_prob = topk_probs[i].item() * parent_prob
                        if tok_prob < prob_threshold:
                            continue

                        tok = topk_tokens[i]
                        current_idx = total_tokens[b]

                        all_tokens[b].append(tok)
                        all_probs[b].append(probs)
                        all_parents[b].append(parent_idx)
                        total_tokens[b] += 1

                        # Prepare features for next level
                        if depth < max_depth - 1:
                            tok_embed = self.token_embed(tok)  # [head_dim]
                            fused = self.fusion(
                                torch.cat([node_features, tok_embed], dim=-1)
                            )
                            next_level[b].append((fused, current_idx, tok_prob))

            current_level = next_level

            # Check if all batches exhausted their budget
            if all(not level for level in current_level):
                break

        # Pad to maximum length and build output tensors
        max_len = max(total_tokens) if total_tokens else 1
        max_len = max(max_len, 1)  # At least 1

        tree_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        tree_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=self.device)

        for b in range(batch_size):
            n = len(all_tokens[b])
            if n == 0:
                # Fallback: single uniform token
                tree_tokens[b, 0] = 0
                tree_mask[b, 0, 0] = True
                continue

            for i, tok in enumerate(all_tokens[b]):
                tree_tokens[b, i] = tok

            # Build tree mask: position i can attend to its ancestors and itself
            for i in range(n):
                # Self attention
                tree_mask[b, i, i] = True

                # Walk up the parent chain
                parent = all_parents[b][i]
                while parent >= 0:
                    tree_mask[b, i, parent] = True
                    if parent < len(all_parents[b]):
                        parent = all_parents[b][parent]
                    else:
                        break

        return tree_tokens, tree_mask

    def build_tree_draft_output(
        self,
        hidden_states: Tensor,
        max_depth: int = 5,
        max_width: int = 3,
        prob_threshold: float = 0.01,
    ) -> TreeDraftOutput:
        """Build complete tree draft output including probabilities.

        Extended version of build_draft_tree that also returns probability
        distributions for each token, suitable for the verifier.

        Args:
            hidden_states: Hidden states from target model, [batch, seq, hidden_size].
            max_depth: Maximum depth of the draft tree.
            max_width: Maximum children per node (top-k branching).
            prob_threshold: Minimum probability to keep a branch.

        Returns:
            TreeDraftOutput with all tree information for verification.
        """
        batch_size = hidden_states.shape[0]
        max_depth = min(max_depth, self._adaptive_depth, len(self.token_heads))
        max_width = min(max_width, self.tree_width)

        h = hidden_states[:, -1, :]
        features = self.feature_extractor(h)

        all_tokens: list[list[Tensor]] = [[] for _ in range(batch_size)]
        all_probs: list[list[Tensor]] = [[] for _ in range(batch_size)]
        all_parents: list[list[int]] = [[] for _ in range(batch_size)]

        current_level: list[list[tuple[Tensor, int, float]]] = [
            [(features[b], -1, 1.0)] for b in range(batch_size)
        ]

        total_tokens = [0 for _ in range(batch_size)]
        max_tokens = self.num_draft_tokens

        for depth in range(max_depth):
            if depth >= len(self.token_heads):
                break

            head = self.token_heads[depth]
            next_level: list[list[tuple[Tensor, int, float]]] = [[] for _ in range(batch_size)]

            for b in range(batch_size):
                if total_tokens[b] >= max_tokens:
                    continue

                for node_features, parent_idx, parent_prob in current_level[b]:
                    if total_tokens[b] >= max_tokens:
                        break

                    logits = head(node_features.unsqueeze(0)).squeeze(0)
                    probs = torch.softmax(logits, dim=-1)
                    topk_probs, topk_tokens = probs.topk(max_width)

                    for i in range(max_width):
                        if total_tokens[b] >= max_tokens:
                            break

                        tok_prob = topk_probs[i].item() * parent_prob
                        if tok_prob < prob_threshold:
                            continue

                        tok = topk_tokens[i]
                        current_idx = total_tokens[b]

                        all_tokens[b].append(tok)
                        all_probs[b].append(probs)
                        all_parents[b].append(parent_idx)
                        total_tokens[b] += 1

                        if depth < max_depth - 1:
                            tok_embed = self.token_embed(tok)
                            fused = self.fusion(
                                torch.cat([node_features, tok_embed], dim=-1)
                            )
                            next_level[b].append((fused, current_idx, tok_prob))

            current_level = next_level
            if all(not level for level in current_level):
                break

        max_len = max(total_tokens) if total_tokens else 1
        max_len = max(max_len, 1)

        tree_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        tree_probs = torch.zeros(
            batch_size, max_len, self.vocab_size,
            dtype=torch.float32, device=self.device
        )
        tree_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=self.device)
        tree_indices = torch.full((batch_size, max_len), -1, dtype=torch.long, device=self.device)

        path_counts = []

        for b in range(batch_size):
            n = len(all_tokens[b])
            if n == 0:
                tree_tokens[b, 0] = 0
                tree_probs[b, 0] = 1.0 / self.vocab_size
                tree_mask[b, 0, 0] = True
                path_counts.append(1)
                continue

            for i, (tok, prob) in enumerate(zip(all_tokens[b], all_probs[b])):
                tree_tokens[b, i] = tok
                tree_probs[b, i] = prob
                tree_indices[b, i] = all_parents[b][i]

            # Count paths (leaf nodes)
            children_count = [0] * n
            for i, parent in enumerate(all_parents[b]):
                if parent >= 0:
                    children_count[parent] += 1
            num_leaves = sum(1 for c in children_count if c == 0)
            path_counts.append(num_leaves)

            for i in range(n):
                tree_mask[b, i, i] = True
                parent = all_parents[b][i]
                while parent >= 0:
                    tree_mask[b, i, parent] = True
                    if parent < len(all_parents[b]):
                        parent = all_parents[b][parent]
                    else:
                        break

        max_paths = max(path_counts)
        path_lengths = torch.zeros(batch_size, max_paths, dtype=torch.long, device=self.device)

        for b in range(batch_size):
            n = len(all_tokens[b]) if all_tokens[b] else 1
            children_count = [0] * n
            for i in range(n):
                parent = all_parents[b][i] if i < len(all_parents[b]) else -1
                if parent >= 0:
                    children_count[parent] += 1

            path_idx = 0
            for i in range(n):
                if children_count[i] == 0:  # Leaf node
                    # Count path length
                    length = 1
                    parent = all_parents[b][i] if i < len(all_parents[b]) else -1
                    while parent >= 0:
                        length += 1
                        if parent < len(all_parents[b]):
                            parent = all_parents[b][parent]
                        else:
                            break
                    if path_idx < max_paths:
                        path_lengths[b, path_idx] = length
                        path_idx += 1

        return TreeDraftOutput(
            tokens=tree_tokens,
            probs=tree_probs,
            tree_mask=tree_mask,
            tree_indices=tree_indices,
            path_lengths=path_lengths,
        )

    def update_acceptance_rate(self, acceptance_rate: float) -> None:
        """Update acceptance tracking for adaptive depth adjustment.

        High acceptance rates allow deeper speculation; low rates trigger
        shallower trees to avoid wasted computation.

        Args:
            acceptance_rate: Fraction of draft tokens accepted in last step.
        """
        self._acceptance_history.append(acceptance_rate)

        # Keep only last 10 steps
        if len(self._acceptance_history) > 10:
            self._acceptance_history = self._acceptance_history[-10:]

        avg_rate = sum(self._acceptance_history) / len(self._acceptance_history)

        # Adaptive depth: increase if acceptance > 70%, decrease if < 30%
        if avg_rate > 0.7 and self._adaptive_depth < self.max_depth:
            self._adaptive_depth = min(self._adaptive_depth + 1, self.max_depth)
        elif avg_rate < 0.3 and self._adaptive_depth > 1:
            self._adaptive_depth = max(self._adaptive_depth - 1, 1)

    def reset(self) -> None:
        """Reset internal state for a new sequence."""
        self._acceptance_history.clear()
        self._adaptive_depth = self.max_depth

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        hidden_size: int,
        vocab_size: int,
        device: torch.device | None = None,
        **kwargs,
    ) -> EagleHead:
        """Load a pretrained Eagle head from a checkpoint.

        Args:
            model_path: Path to the checkpoint file.
            hidden_size: Hidden size of the target model.
            vocab_size: Vocabulary size.
            device: Device to load the model to.
            **kwargs: Additional arguments for EagleHead constructor.

        Returns:
            Loaded EagleHead instance.
        """
        head = cls(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            device=device,
            **kwargs,
        )

        state_dict = torch.load(model_path, map_location=device or "cpu", weights_only=True)
        head.feature_extractor.load_state_dict(
            {k.replace("feature_extractor.", ""): v
             for k, v in state_dict.items() if k.startswith("feature_extractor.")}
        )
        head.token_heads.load_state_dict(
            {k.replace("token_heads.", ""): v
             for k, v in state_dict.items() if k.startswith("token_heads.")}
        )
        head.token_embed.load_state_dict(
            {k.replace("token_embed.", ""): v
             for k, v in state_dict.items() if k.startswith("token_embed.")}
        )
        head.fusion.load_state_dict(
            {k.replace("fusion.", ""): v
             for k, v in state_dict.items() if k.startswith("fusion.")}
        )

        return head
