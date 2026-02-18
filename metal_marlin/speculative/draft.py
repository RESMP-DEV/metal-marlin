"""
Draft model interface for speculative decoding.

Speculative decoding (Leviathan et al., "Fast Inference from Transformers
via Speculative Decoding", ICML 2023) uses a small draft model to propose
tokens that a larger target model verifies in a single forward pass.

The draft model generates k candidate tokens autoregressively (cheap),
then the target model scores all k+1 positions in one pass (parallelized).
Accepted tokens skip target-model decode steps; rejected tokens fall back
to the target model's own prediction at the rejection point.

This module defines the abstract DraftModel interface and three concrete
implementations:
- SmallModelDraft: uses a smaller transformer for drafting
- NGramDraft: uses online n-gram statistics for zero-cost drafting
- EagleHead: uses EAGLE v3 tree attention for high-acceptance drafting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..kv_cache import KVCache

if TYPE_CHECKING:
    pass


class CausalLMDraft(Protocol):
    """Protocol for models usable as draft models."""

    def __call__(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor: ...
    def create_kv_cache(self) -> KVCache: ...


@dataclass
class DraftOutput:
    """Output from a draft model's speculative generation.

    Attributes:
        tokens: Proposed token IDs, shape [batch, num_speculative].
        probs: Full probability distributions for each proposed position,
            shape [batch, num_speculative, vocab_size]. Used by the verifier
            to compute acceptance probabilities.
    """

    tokens: Tensor  # [batch, num_speculative]
    probs: Tensor  # [batch, num_speculative, vocab_size]


class DraftModel(ABC):
    """Abstract interface for draft models in speculative decoding.

    Implementations must produce both token predictions and their full
    probability distributions. The distributions are needed for the
    modified rejection sampling used in verification.
    """

    @abstractmethod
    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative token proposals.

        Args:
            input_ids: Current context token IDs, shape [batch, seq_len].
                For decode-phase usage this is typically [batch, 1] containing
                the last accepted token.
            kv_cache: Target model's KV cache (for context). Draft models
                with their own caches may ignore this.
            num_tokens: Number of tokens to speculatively generate.

        Returns:
            DraftOutput with proposed tokens and their probability distributions.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (caches, statistics) for a new sequence."""
        ...


class SmallModelDraft(DraftModel):
    """Use a smaller causal LM (e.g., 1B params) for drafting.

    The draft model runs its own KV cache independently of the target.
    On each call to speculate(), it generates num_tokens autoregressively
    using greedy decoding (argmax) for maximum throughput.

    Implementation:
    ---------------
    The core generation loop (in speculate() method) is a standard autoregressive
    transformer decode loop, optimized for speed:
      - Uses KV caching to avoid recomputing attention for previous positions
      - Employs greedy sampling (argmax) rather than stochastic sampling
      - Generates one token per iteration, feeding it back as input
      - Maintains probability distributions needed for verification

    The draft model's cache is synchronized: after the target model accepts
    n tokens, the caller should advance the draft cache by feeding those
    tokens or simply reset and re-prefill.

    Performance characteristics:
    - Each draft token costs ~1/10th of a target token (for 10x smaller model)
    - Typical acceptance rate: 60-80% for well-matched draft/target pairs
    - Overall speedup: 2-4x depending on acceptance rate and model size ratio
    """

    def __init__(
        self,
        model: CausalLMDraft,
        max_speculative: int = 4,
        device: torch.device | None = None,
        cache: KVCache | None = None,
    ):
        self.model = model
        self.max_speculative = max_speculative
        self._cache: KVCache | None = cache
        self._cache_seq_len: int = 0
        self.device = device or torch.device("cpu")

    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate K speculative tokens via autoregressive draft model generation loop.

        This implements the core draft model generation loop for speculative decoding.
        The loop runs the small draft model K times autoregressively (with KV caching),
        where each iteration:
          1. Forward pass through draft model to get logits
          2. Extract last position logits (predicting next token)
          3. Compute probabilities via softmax
          4. Sample/select next token (greedy argmax for throughput)
          5. Advance draft model's KV cache
          6. Use the new token as input for next iteration

        The K generated tokens are then verified by the target model in a single
        forward pass, amortizing the expensive computation.

        Greedy decoding is used (argmax) rather than sampling because:
        - The verifier will correct any mismatches via rejection sampling
        - Greedy maximizes acceptance rate for well-matched draft/target pairs
        - Sampling would add variance without improving the final distribution

        Args:
            input_ids: Current context token IDs, shape [batch, seq_len].
                Typically [batch, 1] containing the last accepted token in decode phase.
            kv_cache: Target model's KV cache (unused; draft maintains its own cache).
            num_tokens: Number of speculative tokens to generate (K in the literature).

        Returns:
            DraftOutput with:
              - tokens: K proposed token IDs [batch, K]
              - probs: K probability distributions [batch, K, vocab_size]
                Used by the verifier for rejection sampling calculations.
        """
        num_tokens = min(num_tokens, self.max_speculative)
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize draft cache if needed
        if self._cache is None:
            self._cache = self.model.create_kv_cache()

        # Get vocab size for pre-allocation (cache on first call)
        vocab_size: int
        if hasattr(self.model, "vocab_size"):
            vocab_size = self.model.vocab_size  # type: ignore[union-attr]
        else:
            # Infer from first forward pass
            with torch.inference_mode():
                first_logits = self.model(input_ids, kv_cache=self._cache)
                vocab_size = first_logits.shape[-1]

        # Pre-allocate output tensors on the correct device for better performance
        # Avoids repeated list append and torch.stack overhead
        tokens = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)
        probs = torch.zeros((batch_size, num_tokens, vocab_size), dtype=torch.float32, device=device)

        # Pre-allocate input buffer for next token to avoid repeated allocations
        next_input = torch.empty((batch_size, 1), dtype=torch.long, device=device)

        # === DRAFT MODEL GENERATION LOOP ===
        # Autoregressive loop: generate K tokens one at a time using draft model
        # Use inference_mode for maximum performance (disables all gradient-related tracking)
        current_ids = input_ids
        with torch.inference_mode():
            for step in range(num_tokens):
                # Step 1: Forward pass through draft model
                # Returns logits [batch, current_seq_len, vocab_size]
                logits = self.model(current_ids, kv_cache=self._cache)
                
                # Step 2: Extract logits for last position (predicting next token)
                # Shape: [batch, vocab_size]
                last_logits = logits[:, -1, :]

                # Step 3: Convert logits to probability distribution
                # Use in-place softmax when possible for memory efficiency
                next_probs = torch.softmax(last_logits, dim=-1)
                
                # Step 4: Greedy selection for maximum acceptance rate
                # Shape: [batch]
                next_token = torch.argmax(next_probs, dim=-1)

                # Store results directly into pre-allocated tensors
                tokens[:, step] = next_token
                probs[:, step, :] = next_probs

                # Step 5: Advance draft model's KV cache to include processed positions
                self._cache.advance(current_ids.shape[1])
                
                # Step 6: Prepare input for next iteration using pre-allocated buffer
                # Copy token into pre-allocated buffer to avoid new tensor creation
                next_input.copy_(next_token.unsqueeze(1))
                current_ids = next_input
        # === END GENERATION LOOP ===

        return DraftOutput(tokens=tokens, probs=probs)

    def reset(self) -> None:
        """Reset the draft model's KV cache for a new sequence."""
        if self._cache is not None:
            self._cache.reset()
            self._cache = None
        self._cache_seq_len = 0

    def sync_after_accept(self, accepted_tokens: Tensor) -> None:
        """Feed accepted tokens into draft cache to keep it synchronized.

        After the target model accepts some subset of the draft's proposals,
        call this so the draft cache reflects the true sequence state.

        In practice it's often simpler to reset and re-prefill the draft
        model, but for long sequences this incremental sync saves compute.

        Args:
            accepted_tokens: The tokens accepted by the target, [batch, n_accepted].
        """
        if self._cache is None:
            self._cache = self.model.create_kv_cache()
        # Run accepted tokens through draft model to populate its cache
        self.model(accepted_tokens, kv_cache=self._cache)
        self._cache.advance(accepted_tokens.shape[1])


class NGramDraft(DraftModel):
    """Use online n-gram statistics for zero-cost speculative drafting.

    This approach requires no additional model parameters. It builds n-gram
    frequency tables from the generated text so far, then uses the most
    frequent continuations as draft proposals.

    Works best for repetitive or formulaic text (code, structured data).
    Falls back to uniform distributions when no n-gram match exists.

    The probability estimates are crude (based on count ratios) but sufficient
    for the verifier's rejection sampling since mismatches just result in
    rejection with fallback to the target model's own prediction.
    """

    def __init__(
        self, ngram_size: int = 3, vocab_size: int = 32000, device: torch.device | None = None
    ):
        self.ngram_size = ngram_size
        self.vocab_size = vocab_size
        self.ngram_counts: dict[tuple[int, ...], dict[int, int]] = {}
        self._history: list[int] = []
        self.device = device or torch.device("cpu")

    def update_ngrams(self, token_ids: list[int]) -> None:
        """Update n-gram statistics from newly generated tokens.

        Should be called after each verification step with the accepted tokens.

        Args:
            token_ids: Sequence of token IDs to learn from.
        """
        self._history.extend(token_ids)
        # Build n-grams from the extended history
        for i in range(
            max(0, len(self._history) - self.ngram_size - len(token_ids)),
            len(self._history) - self.ngram_size,
        ):
            context = tuple(self._history[i : i + self.ngram_size])
            next_token = self._history[i + self.ngram_size]
            if context not in self.ngram_counts:
                self.ngram_counts[context] = {}
            self.ngram_counts[context][next_token] = (
                self.ngram_counts[context].get(next_token, 0) + 1
            )

    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        batch_size = input_ids.shape[0]

        # Extract the last ngram_size tokens from input as initial context
        seq_len = input_ids.shape[1]
        if seq_len < self.ngram_size:
            # Not enough context; return uniform guesses (token 0)
            return self._uniform_fallback(batch_size, num_tokens)

        # Work with first batch element (n-gram draft is not naturally batched)
        context_ids = input_ids[0, -self.ngram_size :].tolist()

        tokens: list[int] = []
        probs_list: list[Tensor] = []

        for _ in range(num_tokens):
            context = tuple(context_ids[-self.ngram_size :])
            counts = self.ngram_counts.get(context)

            if counts:
                # Build probability distribution from counts
                total = sum(counts.values())
                prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
                for tok, count in counts.items():
                    if tok < self.vocab_size:
                        prob_dist[tok] = count / total

                # Pick the most likely token
                best_token = max(counts, key=counts.get)  # type: ignore[arg-type]
                if best_token >= self.vocab_size:
                    best_token = 0
                    prob_dist = (
                        torch.ones(self.vocab_size, dtype=torch.float32, device=self.device)
                        / self.vocab_size
                    )
            else:
                # No n-gram match: uniform distribution, predict token 0
                prob_dist = (
                    torch.ones(self.vocab_size, dtype=torch.float32, device=self.device)
                    / self.vocab_size
                )
                best_token = 0

            tokens.append(best_token)
            probs_list.append(prob_dist)

            # Extend context for next prediction
            context_ids.append(best_token)

        # Stack into batch-compatible shapes
        tokens_arr = torch.tensor(tokens, dtype=torch.long, device=self.device).reshape(
            1, num_tokens
        )
        probs_arr = torch.stack(probs_list, dim=0).reshape(1, num_tokens, self.vocab_size)

        # Broadcast to batch dimension if needed
        if batch_size > 1:
            tokens_arr = tokens_arr.expand(batch_size, num_tokens)
            probs_arr = probs_arr.expand(batch_size, num_tokens, self.vocab_size)

        return DraftOutput(tokens=tokens_arr, probs=probs_arr)

    def reset(self) -> None:
        """Clear n-gram statistics and history for a new sequence."""
        self.ngram_counts.clear()
        self._history.clear()

    def _uniform_fallback(self, batch_size: int, num_tokens: int) -> DraftOutput:
        """Return uniform distributions when no context is available."""
        tokens = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=self.device)
        probs = (
            torch.ones(1, 1, self.vocab_size, dtype=torch.float32, device=self.device)
            / self.vocab_size
        ).expand(batch_size, num_tokens, self.vocab_size)
        return DraftOutput(tokens=tokens, probs=probs)


# ---------------------------------------------------------------------------
# EAGLE v3 Tree Attention Draft Model
# ---------------------------------------------------------------------------


@dataclass
class EagleTreeNode:
    """Node in the EAGLE speculation tree.

    Each node represents a token in the draft tree. Children represent
    alternative continuations explored in parallel via tree attention.

    Attributes:
        token: The token ID at this position (None for root).
        prob: The probability assigned by the draft head.
        logits: Full logit distribution at this position.
        depth: Distance from root (0 for root node).
        children: List of child nodes (alternative continuations).
        parent: Parent node (None for root).
    """

    token: int | None = None
    prob: float = 1.0
    logits: Tensor | None = None
    depth: int = 0
    children: list[EagleTreeNode] = field(default_factory=list)
    parent: EagleTreeNode | None = None


@dataclass
class EagleTreeOutput:
    """Output from EAGLE tree speculation.

    Contains both the tree structure and flattened tensors suitable for
    batched verification with the target model.

    Attributes:
        root: Root node of the speculation tree.
        flat_tokens: Flattened token sequences for batched verification,
            shape [num_paths, max_depth].
        flat_probs: Flattened probability distributions,
            shape [num_paths, max_depth, vocab].
        path_lengths: Number of valid tokens in each path, shape [num_paths].
        tree_attention_mask: Attention mask encoding tree structure for
            efficient parallel verification.
    """

    root: EagleTreeNode
    flat_tokens: Tensor  # [num_paths, max_depth]
    flat_probs: Tensor  # [num_paths, max_depth, vocab]
    path_lengths: Tensor  # [num_paths]
    tree_attention_mask: Tensor | None = None  # [num_paths, seq_len, seq_len]


class EagleHeadModule(nn.Module):
    """Lightweight transformer head for EAGLE drafting.

    Takes hidden states from the target model and predicts next-token
    distributions. Trained to match the target model's predictions while
    being much cheaper to run.

    Architecture:
    - Input projection (optional, if hidden dims differ)
    - 1-2 transformer layers with reduced width
    - Output projection to vocabulary
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        intermediate_size: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        intermediate_size = intermediate_size or hidden_size * 2

        # Token embedding for draft continuation
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Fusion layer: combine target hidden state with draft token embedding
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)

        # Lightweight transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=intermediate_size,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        draft_token: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass predicting next-token logits.

        Args:
            hidden_states: Target model hidden states, [batch, seq, hidden].
            draft_token: Previously drafted token for autoregressive drafting,
                [batch] or None for first draft position.
            attention_mask: Optional attention mask for tree attention.

        Returns:
            Logits for next token, [batch, vocab].
        """
        # Take last position's hidden state
        h = hidden_states[:, -1:, :]  # [batch, 1, hidden]

        # Fuse with draft token embedding if provided
        if draft_token is not None:
            tok_embed = self.embed_tokens(draft_token.reshape(-1, 1))  # [batch, 1, hidden]
            h = torch.cat([h, tok_embed], dim=-1)  # [batch, 1, hidden*2]
            h = self.fusion(h)  # [batch, 1, hidden]

        # Run through transformer layers
        for layer in self.layers:
            h = layer(h, src_mask=attention_mask)

        # Output projection
        h = self.norm(h)
        logits = self.lm_head(h)  # [batch, 1, vocab]

        return logits.squeeze(1)  # [batch, vocab]


class EagleHead(DraftModel):
    """EAGLE v3 draft model using tree attention for speculative decoding.

    EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
    uses a small draft head attached to the target model's hidden states.
    Key features:

    1. **Tree speculation**: Instead of linear speculation, EAGLE explores
       multiple continuation paths in parallel, forming a tree structure.
       This increases the probability that at least one path matches the
       target model's preferences.

    2. **Hidden state drafting**: The draft head takes the target model's
       hidden states as input, not just the previous token. This provides
       richer context and improves prediction accuracy.

    3. **Adaptive tree width**: The tree width can adapt based on observed
       acceptance rates - wider trees when acceptance is low, narrower when
       high.

    Usage:
        # Create from a target model
        eagle = EagleHead.from_target_model(
            target_model,
            tree_width=3,
            max_depth=5,
        )

        # Or with custom head
        head = EagleHeadModule(hidden_size=4096, vocab_size=32000)
        eagle = EagleHead(head, vocab_size=32000, tree_width=3)

        # Draft tokens
        output = eagle.speculate(input_ids, num_tokens=4)
    """

    def __init__(
        self,
        head: EagleHeadModule | None = None,
        vocab_size: int = 32000,
        tree_width: int = 3,
        max_depth: int = 5,
        temperature: float = 1.0,
        top_k: int = 10,
        adaptive_width: bool = True,
        min_width: int = 1,
        max_width: int = 5,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize EAGLE draft model.

        Args:
            head: Pre-trained EagleHeadModule. If None, must call
                from_target_model() or set_head() before use.
            vocab_size: Vocabulary size for the model.
            tree_width: Number of top-k candidates to explore at each node.
            max_depth: Maximum depth of the speculation tree.
            temperature: Sampling temperature for draft predictions.
            top_k: Number of candidates to consider at each position.
            adaptive_width: If True, adjust tree width based on acceptance.
            min_width: Minimum tree width when adaptive.
            max_width: Maximum tree width when adaptive.
            device: Device for tensors.
            dtype: Data type for computations.
        """
        self.head = head
        self.vocab_size = vocab_size
        self.tree_width = tree_width
        self.max_depth = max_depth
        self.temperature = temperature
        self.top_k = top_k
        self.adaptive_width = adaptive_width
        self.min_width = min_width
        self.max_width = max_width
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Adaptive state
        self._acceptance_history: list[float] = []
        self._current_width = tree_width

        # Cached hidden states from target model
        self._cached_hidden: Tensor | None = None
        self._cached_position: int = 0

    @classmethod
    def from_target_model(
        cls,
        target_model: Any,
        tree_width: int = 3,
        max_depth: int = 5,
        num_layers: int = 1,
        **kwargs: Any,
    ) -> EagleHead:
        """Create an EAGLE head from a target model's configuration.

        Initializes a draft head with dimensions matching the target model.
        The head weights are initialized randomly and should be fine-tuned
        on target model outputs for best performance.

        Args:
            target_model: Target language model with config attribute.
            tree_width: Number of candidates per tree node.
            max_depth: Maximum speculation tree depth.
            num_layers: Number of transformer layers in draft head.
            **kwargs: Additional arguments passed to EagleHead.__init__.

        Returns:
            Configured EagleHead instance.
        """
        # Extract model configuration
        config = getattr(target_model, "config", None)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", 4096)
            vocab_size = getattr(config, "vocab_size", 32000)
            num_heads = getattr(config, "num_attention_heads", 32)
            # Reduce head count for draft model
            draft_heads = max(4, num_heads // 4)
        else:
            # Fallback defaults
            hidden_size = 4096
            vocab_size = 32000
            draft_heads = 8

        device = kwargs.pop("device", None)
        if device is None:
            # Try to infer from target model
            try:
                device = next(target_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        head = EagleHeadModule(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=draft_heads,
        )
        head = head.to(device)

        return cls(
            head=head,
            vocab_size=vocab_size,
            tree_width=tree_width,
            max_depth=max_depth,
            device=device,
            **kwargs,
        )

    def set_hidden_states(self, hidden_states: Tensor, position: int = 0) -> None:
        """Cache hidden states from the target model for drafting.

        Should be called after each target model forward pass to provide
        context for the draft head.

        Args:
            hidden_states: Hidden states from target model, [batch, seq, hidden].
            position: Sequence position of the hidden states.
        """
        self._cached_hidden = hidden_states
        self._cached_position = position

    def speculate(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative token proposals using tree exploration.

        Builds a tree of candidate continuations, then flattens the best
        paths into a format suitable for batched verification.

        Args:
            input_ids: Current context tokens, [batch, seq_len].
            kv_cache: Target model's KV cache (contains hidden states).
            num_tokens: Target number of tokens per path (max_depth).

        Returns:
            DraftOutput with proposed tokens and probabilities.
            For tree speculation, returns the best single path.
            Use speculate_tree() for full tree output.
        """
        batch_size = input_ids.shape[0]
        num_tokens = min(num_tokens, self.max_depth)

        if self.head is None:
            # No head available, fall back to uniform prediction
            return self._fallback_output(batch_size, num_tokens)

        # Get hidden states - use cached if available
        if self._cached_hidden is not None:
            hidden = self._cached_hidden
        else:
            # No cached hidden states - need to run through target model first
            # For now, fall back to simpler prediction
            return self._fallback_output(batch_size, num_tokens)

        # Build speculation tree
        tree = self._build_tree(hidden, num_tokens)

        # Extract best path from tree
        best_path = self._extract_best_path(tree)

        # Convert to DraftOutput format
        tokens = torch.tensor(
            [best_path["tokens"][:num_tokens]],
            dtype=torch.long,
            device=self.device,
        )
        probs = torch.stack(best_path["probs"][:num_tokens], dim=0).unsqueeze(0)

        # Pad if needed
        if tokens.shape[1] < num_tokens:
            pad_len = num_tokens - tokens.shape[1]
            tokens = F.pad(tokens, (0, pad_len), value=0)
            uniform = torch.ones(1, pad_len, self.vocab_size, device=self.device) / self.vocab_size
            probs = torch.cat([probs, uniform], dim=1)

        # Broadcast to batch size
        if batch_size > 1:
            tokens = tokens.expand(batch_size, -1)
            probs = probs.expand(batch_size, -1, -1)

        return DraftOutput(tokens=tokens, probs=probs)

    def speculate_tree(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> EagleTreeOutput:
        """Generate full speculation tree for tree-based verification.

        Returns the complete tree structure along with flattened tensors
        for efficient batched verification.

        Args:
            input_ids: Current context tokens, [batch, seq_len].
            kv_cache: Target model's KV cache.
            num_tokens: Target tree depth.

        Returns:
            EagleTreeOutput with tree structure and flattened verification data.
        """
        if self.head is None or self._cached_hidden is None:
            # Return degenerate tree
            root = EagleTreeNode()
            return EagleTreeOutput(
                root=root,
                flat_tokens=torch.zeros(1, num_tokens, dtype=torch.long, device=self.device),
                flat_probs=torch.ones(1, num_tokens, self.vocab_size, device=self.device)
                / self.vocab_size,
                path_lengths=torch.ones(1, dtype=torch.int32, device=self.device),
            )

        tree = self._build_tree(self._cached_hidden, num_tokens)
        return self._flatten_tree(tree, num_tokens)

    def _build_tree(self, hidden: Tensor, depth: int) -> EagleTreeNode:
        """Build speculation tree via breadth-first expansion.

        At each node, generates top-k candidate continuations and adds
        them as children. The tree width is controlled by tree_width.
        """
        width = self._current_width if self.adaptive_width else self.tree_width
        root = EagleTreeNode(depth=0)

        # BFS queue: (node, hidden_state, draft_token)
        queue: list[tuple[EagleTreeNode, Tensor, Tensor | None]] = [(root, hidden, None)]

        while queue:
            node, h, prev_token = queue.pop(0)

            if node.depth >= depth:
                continue

            # Get logits from draft head
            with torch.no_grad():
                logits = self.head(h, draft_token=prev_token)  # [batch, vocab]

            # Apply temperature
            if self.temperature > 0:
                scaled_logits = logits / self.temperature
            else:
                scaled_logits = logits

            probs = F.softmax(scaled_logits, dim=-1)

            # Get top-k candidates
            top_probs, top_tokens = torch.topk(probs, min(width, self.top_k), dim=-1)

            # Create children for top candidates
            for i in range(min(width, top_tokens.shape[-1])):
                tok = int(top_tokens[0, i].item())
                prob = float(top_probs[0, i].item())

                child = EagleTreeNode(
                    token=tok,
                    prob=prob,
                    logits=logits[0],
                    depth=node.depth + 1,
                    parent=node,
                )
                node.children.append(child)

                # Update hidden state for this path
                tok_tensor = torch.tensor([tok], dtype=torch.long, device=self.device)

                # For deeper nodes, continue expansion
                if child.depth < depth:
                    queue.append((child, h, tok_tensor))

        return root

    def _extract_best_path(self, root: EagleTreeNode) -> dict[str, list]:
        """Extract the highest-probability path from the tree."""
        best_path: dict[str, list] = {"tokens": [], "probs": []}
        best_prob = 0.0

        def dfs(node: EagleTreeNode, path: list[int], prob_list: list[Tensor], cum_prob: float):
            nonlocal best_path, best_prob

            if not node.children:
                # Leaf node - check if this is the best path
                if cum_prob > best_prob:
                    best_prob = cum_prob
                    best_path = {"tokens": path.copy(), "probs": prob_list.copy()}
                return

            for child in node.children:
                if child.token is not None:
                    new_path = path + [child.token]
                    new_prob = cum_prob * child.prob

                    # Create probability distribution for this position
                    if child.logits is not None:
                        probs = F.softmax(child.logits, dim=-1)
                    else:
                        probs = torch.ones(self.vocab_size, device=self.device) / self.vocab_size

                    new_prob_list = prob_list + [probs]
                    dfs(child, new_path, new_prob_list, new_prob)

        dfs(root, [], [], 1.0)
        return best_path

    def _flatten_tree(self, root: EagleTreeNode, max_depth: int) -> EagleTreeOutput:
        """Flatten tree into batched tensors for verification."""
        paths: list[list[int]] = []
        probs_list: list[list[Tensor]] = []

        def collect_paths(node: EagleTreeNode, path: list[int], probs: list[Tensor]):
            if not node.children or len(path) >= max_depth:
                if path:  # Skip empty paths
                    paths.append(path)
                    probs_list.append(probs)
                return

            for child in node.children:
                if child.token is not None:
                    new_path = path + [child.token]
                    child_probs = (
                        F.softmax(child.logits, dim=-1)
                        if child.logits is not None
                        else torch.ones(self.vocab_size, device=self.device) / self.vocab_size
                    )
                    new_probs = probs + [child_probs]
                    collect_paths(child, new_path, new_probs)

        collect_paths(root, [], [])

        if not paths:
            # Empty tree - return single uniform path
            return EagleTreeOutput(
                root=root,
                flat_tokens=torch.zeros(1, max_depth, dtype=torch.long, device=self.device),
                flat_probs=torch.ones(1, max_depth, self.vocab_size, device=self.device)
                / self.vocab_size,
                path_lengths=torch.ones(1, dtype=torch.int32, device=self.device),
            )

        # Pad paths to max_depth
        num_paths = len(paths)
        flat_tokens = torch.zeros(num_paths, max_depth, dtype=torch.long, device=self.device)
        flat_probs = (
            torch.ones(num_paths, max_depth, self.vocab_size, device=self.device) / self.vocab_size
        )
        path_lengths = torch.zeros(num_paths, dtype=torch.int32, device=self.device)

        for i, (path, probs) in enumerate(zip(paths, probs_list)):
            path_len = len(path)
            path_lengths[i] = path_len
            flat_tokens[i, :path_len] = torch.tensor(path, dtype=torch.long, device=self.device)
            for j, p in enumerate(probs):
                flat_probs[i, j] = p

        return EagleTreeOutput(
            root=root,
            flat_tokens=flat_tokens,
            flat_probs=flat_probs,
            path_lengths=path_lengths,
        )

    def _fallback_output(self, batch_size: int, num_tokens: int) -> DraftOutput:
        """Generate fallback output when head is not available."""
        tokens = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=self.device)
        probs = (
            torch.ones(batch_size, num_tokens, self.vocab_size, device=self.device) / self.vocab_size
        )
        return DraftOutput(tokens=tokens, probs=probs)

    def update_acceptance(self, acceptance_rate: float) -> None:
        """Update adaptive tree width based on acceptance rate.

        Higher acceptance rates allow narrower trees (faster).
        Lower acceptance rates trigger wider trees (more exploration).

        Args:
            acceptance_rate: Fraction of draft tokens accepted in last step.
        """
        if not self.adaptive_width:
            return

        self._acceptance_history.append(acceptance_rate)

        # Use recent history for adaptation
        window = self._acceptance_history[-10:]
        avg_rate = sum(window) / len(window)

        # Adapt tree width
        if avg_rate < 0.3:
            # Low acceptance - widen tree for more exploration
            self._current_width = min(self._current_width + 1, self.max_width)
        elif avg_rate > 0.7:
            # High acceptance - narrow tree for speed
            self._current_width = max(self._current_width - 1, self.min_width)

    def reset(self) -> None:
        """Reset internal state for a new sequence."""
        self._cached_hidden = None
        self._cached_position = 0
        self._acceptance_history.clear()
        self._current_width = self.tree_width
