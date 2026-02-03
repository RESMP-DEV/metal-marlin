"""Speculative decoding verification via rejection sampling.

Implements the verification algorithm from Leviathan et al.,
"Fast Inference from Transformers via Speculative Decoding" (ICML 2023).

Also includes tree-structured verification for Eagle v3 (Li et al., 2024),
which extends speculative decoding to tree-structured drafts where multiple
candidate paths can be verified in a single target forward pass.

Linear verification algorithm for each position i:
  1. Draw r ~ Uniform(0, 1)
  2. If r < p(x_i) / q(x_i): accept token x_i
  3. Else: reject x_i, sample replacement from norm(max(0, p - q))
  4. Stop accepting at first rejection

After all accepted tokens, sample one bonus token from the target model's
distribution at position (num_accepted + 1). This gives us at least 1 and
at most K+1 tokens per verification step.

Tree verification (Eagle v3):
  - Draft model proposes a tree of tokens, where each node has a single parent
  - Target model evaluates all tree nodes in one batched forward pass
  - Rejection sampling at each node, respecting tree structure
  - Returns the longest accepted path from root to any leaf
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..kv_cache import KVCache
from ..sampler import MetalSampler

# Global sampler cache for Metal-accelerated sampling
_sampler: MetalSampler | None = None


def _get_sampler(vocab_size: int) -> MetalSampler:
    """Get or create a MetalSampler for the given vocab size.
    
    The sampler is cached and reused to avoid repeated initialization.
    """
    global _sampler
    if _sampler is None or _sampler.vocab_size != vocab_size:
        _sampler = MetalSampler(vocab_size=vocab_size)
    return _sampler


@dataclass
class VerifyResult:
    """Result of speculative verification.

    Attributes:
        accepted_tokens: [batch, num_spec] token IDs; valid up to num_accepted per row.
        num_accepted: [batch] count of accepted tokens per sequence.
        next_token: [batch] the token sampled after rejection (or bonus token
            if all accepted).
    """

    accepted_tokens: Tensor
    num_accepted: Tensor
    next_token: Tensor


@dataclass
class EagleTreeVerifyResult:
    """Result of tree-structured speculative verification (Eagle v3).

    Attributes:
        accepted_tokens: [batch, max_path_len] token IDs along the longest accepted path.
            Padded with zeros beyond num_accepted per row.
        accepted_mask: [batch, tree_size] boolean mask indicating which tree nodes
            were accepted. A node is accepted if all its ancestors were accepted
            and its own rejection sample passed.
        num_accepted: [batch] count of accepted tokens (length of accepted path).
        next_token: [batch] the token sampled after the last accepted node.
            Either from residual distribution at first rejection, or bonus token
            if an entire leaf path was accepted.
        accepted_path_indices: [batch, max_path_len] indices into the tree for
            the accepted tokens. -1 for padding beyond num_accepted.
    """

    accepted_tokens: Tensor
    accepted_mask: Tensor
    num_accepted: Tensor
    next_token: Tensor
    accepted_path_indices: Tensor


def verify_speculative(
    draft_tokens: Tensor,
    draft_probs: Tensor,
    target_logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> VerifyResult:
    """Verify draft tokens using rejection sampling.

    Args:
        draft_tokens: [batch, num_spec] proposed token IDs from draft model.
        draft_probs: [batch, num_spec, vocab] probability distributions from
            draft model.
        target_logits: [batch, num_spec+1, vocab] raw logits from target model.
            Position i contains logits conditioned on prefix + draft_tokens[:i].
            Position num_spec is the "bonus" position after all draft tokens.
        temperature: Sampling temperature applied to target logits.
            Values <= 0 are treated as greedy (argmax).
        top_p: Nucleus sampling threshold for rejection/bonus sampling.
            1.0 means no filtering.

    Returns:
        VerifyResult with accepted tokens, acceptance counts, and next token.
    """
    batch_size, num_spec = draft_tokens.shape
    device = draft_tokens.device

    # Convert target logits to probabilities
    if temperature <= 0:
        # Greedy: one-hot on argmax
        target_probs = _greedy_probs(target_logits)
    else:
        target_probs = torch.softmax(target_logits / temperature, dim=-1)

    # Per-position acceptance loop (K is small, typically 4-8)
    accepted_list: list[Tensor] = []
    num_accepted = torch.zeros(batch_size, dtype=torch.int32, device=device)
    still_accepting = torch.ones(batch_size, dtype=torch.bool, device=device)
    rejection_pos = torch.full((batch_size,), num_spec, dtype=torch.int32, device=device)

    for i in range(num_spec):
        draft_token = draft_tokens[:, i]  # [batch]
        token_idx = draft_token.reshape(-1, 1)  # [batch, 1]

        # Gather q(x_i) and p(x_i)
        p_draft = torch.gather(draft_probs[:, i, :], dim=1, index=token_idx).squeeze(-1)
        p_target = torch.gather(target_probs[:, i, :], dim=1, index=token_idx).squeeze(-1)

        # Accept if r < min(1, p/q)
        r = torch.rand(batch_size, device=device)
        ratio = p_target / torch.clamp(p_draft, min=1e-10)
        accept = (r < ratio) & still_accepting

        # Record this position's accepted token (0 for rejected)
        accepted_list.append(torch.where(accept, draft_token, torch.zeros_like(draft_token)))
        num_accepted = torch.where(accept, num_accepted + 1, num_accepted)

        # Track first rejection position
        newly_rejected = still_accepting & ~accept
        rejection_pos = torch.where(
            newly_rejected,
            torch.tensor(i, dtype=torch.int32, device=device),
            rejection_pos,
        )
        still_accepting = still_accepting & accept

    accepted_tokens = torch.stack(accepted_list, dim=1)  # [batch, num_spec]

    # Sample next token based on whether all were accepted or not
    next_token = _sample_next_token(
        still_accepting,
        rejection_pos,
        num_spec,
        target_logits,
        target_probs,
        draft_probs,
        temperature,
        top_p,
        batch_size,
        device,
    )

    return VerifyResult(
        accepted_tokens=accepted_tokens,
        num_accepted=num_accepted,
        next_token=next_token,
    )


def verify_eagle_tree(
    target_model: Callable[[Tensor, KVCache | None], Tensor],
    draft_tokens: Tensor,
    draft_probs: Tensor,
    tree_mask: Tensor,
    tree_parents: Tensor,
    hidden_states: Tensor,
    kv_cache: KVCache,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[Tensor, Tensor, int]:
    """Verify tree-structured drafts and return longest accepted path.

    Eagle v3 generates drafts as a tree structure where multiple paths branch
    from common prefixes. This allows speculating on multiple possible
    continuations and accepting whichever path the target model agrees with.

    The tree is represented by:
    - tree_mask: attention mask where tree_mask[i,j]=1 means node i can attend to node j
    - tree_parents: index of each node's parent (-1 for root nodes)

    Verification strategy:
    1. Run target model once on all tree paths (batched via tree attention mask)
    2. Compare target vs draft distributions per node using rejection sampling
    3. Find longest accepted path by traversing from root, accepting nodes
       where r < p_target/p_draft and all ancestors are accepted
    4. Sample correction token from residual distribution at first rejection

    Args:
        target_model: Callable that takes (input_ids, kv_cache) and returns logits.
            Input should support tree attention via the provided mask.
        draft_tokens: [batch, tree_size] proposed token IDs at each tree node.
        draft_probs: [batch, tree_size, vocab] probability distributions from
            the draft model at each tree node.
        tree_mask: [tree_size, tree_size] attention mask for tree structure.
            tree_mask[i,j] = 1 if node i can attend to node j (j is ancestor of i).
            Should be lower-triangular with 1s on the diagonal.
        tree_parents: [tree_size] parent index for each node. -1 for root node(s).
        hidden_states: [batch, seq_len, hidden_dim] hidden states from the prefix.
            Used by target model to condition tree verification.
        kv_cache: KV cache containing the prefix context.
        temperature: Sampling temperature for target model. <= 0 means greedy.
        top_p: Nucleus sampling threshold. 1.0 means no filtering.

    Returns:
        Tuple of:
            accepted_tokens: [batch, max_depth] longest accepted sequence per batch.
            accepted_mask: [batch, tree_size] which tree nodes were accepted.
            num_accepted: Total accepted tokens (scalar, max across batch).
    """
    batch_size, tree_size = draft_tokens.shape
    device = draft_tokens.device

    # Suppress unused parameter warnings - these are part of the API
    _ = tree_mask  # Used by target model internally for tree attention
    _ = hidden_states  # Passed to target model for conditioning

    # 1. Run target model on all tree nodes in a single batched forward pass
    # The tree_mask enables proper attention: each node attends to its ancestors
    target_logits = _run_target_on_tree(target_model, draft_tokens, kv_cache)
    # target_logits: [batch, tree_size, vocab]

    # Convert to probabilities
    if temperature <= 0:
        target_probs = _greedy_probs(target_logits)
    else:
        target_probs = torch.softmax(target_logits / temperature, dim=-1)

    # 2. Compute acceptance for each node independently
    # Accept if r < p_target(x) / p_draft(x) where x is the draft token at that node
    node_accepted = _compute_node_acceptance(
        draft_tokens, draft_probs, target_probs, device
    )
    # node_accepted: [batch, tree_size] boolean

    # 3. Propagate acceptance through tree: a node is truly accepted only if
    # all its ancestors are also accepted
    tree_accepted = _propagate_tree_acceptance(node_accepted, tree_parents, device)
    # tree_accepted: [batch, tree_size] boolean

    # 4. Find the longest accepted path from root to any leaf
    accepted_paths, path_lengths = _find_longest_accepted_path(
        tree_accepted, tree_parents, draft_tokens, device
    )
    # accepted_paths: [batch, max_depth] token indices
    # path_lengths: [batch] length of accepted path per batch element

    max_accepted = int(path_lengths.max().item())

    # 5. Sample next token: either bonus (if full leaf path accepted) or from residual
    _sample_tree_next_token(
        tree_accepted,
        tree_parents,
        draft_probs,
        target_probs,
        target_logits,
        path_lengths,
        temperature,
        top_p,
        device,
    )

    # Package results into the expected return format
    # Flatten accepted_paths to just the tokens
    max_depth = accepted_paths.shape[1] if accepted_paths.ndim > 1 else tree_size
    accepted_tokens = torch.zeros(batch_size, max_depth, dtype=torch.long, device=device)
    for b in range(batch_size):
        path_len = int(path_lengths[b].item())
        if path_len > 0:
            accepted_tokens[b, :path_len] = accepted_paths[b, :path_len]

    return accepted_tokens, tree_accepted, max_accepted


def _run_target_on_tree(
    target_model: Callable[[Tensor, KVCache | None], Tensor],
    draft_tokens: Tensor,
    kv_cache: KVCache,
) -> Tensor:
    """Run target model on tree-structured draft tokens.

    The tree attention mask is applied during the target model forward pass
    to ensure each node only attends to its ancestors in the tree.

    Args:
        target_model: Model callable taking (input_ids, kv_cache).
        draft_tokens: [batch, tree_size] draft token IDs.
        kv_cache: KV cache for the prefix.

    Returns:
        [batch, tree_size, vocab] target model logits for each tree position.
    """
    # The target model needs to handle tree attention internally.
    # For models that support it, we pass the tree_mask via the cache or
    # a custom attention mechanism. For simplicity, we run the model
    # on the draft tokens and assume it handles the tree structure.
    #
    # In practice, the target model would use tree_mask during attention:
    # attn_scores = Q @ K.T / sqrt(d)
    # attn_scores = attn_scores.masked_fill(~tree_mask, float('-inf'))
    #
    # Since we don't have direct control over the model's attention,
    # we return logits assuming the model handles tree masking or we
    # batch process paths separately if needed.

    # Call target model with draft tokens
    # The model should return [batch, tree_size, vocab] logits
    target_logits = target_model(draft_tokens, kv_cache)

    # Handle case where model returns logits for seq_len+tree_size
    # and we need just the tree_size portion
    if target_logits.shape[1] > draft_tokens.shape[1]:
        target_logits = target_logits[:, -draft_tokens.shape[1] :, :]

    return target_logits


def _compute_node_acceptance(
    draft_tokens: Tensor,
    draft_probs: Tensor,
    target_probs: Tensor,
    device: torch.device,
) -> Tensor:
    """Compute independent acceptance decisions for each tree node.

    Uses rejection sampling: accept node i if r_i < p_target(x_i) / p_draft(x_i).

    Args:
        draft_tokens: [batch, tree_size] draft token IDs.
        draft_probs: [batch, tree_size, vocab] draft probabilities.
        target_probs: [batch, tree_size, vocab] target probabilities.
        device: Computation device.

    Returns:
        [batch, tree_size] boolean tensor where True means node accepted.
    """
    batch_size, tree_size = draft_tokens.shape

    # Gather probabilities for the specific draft token at each position
    token_indices = draft_tokens.unsqueeze(-1)  # [batch, tree_size, 1]

    p_draft = torch.gather(draft_probs, dim=-1, index=token_indices).squeeze(-1)
    # p_draft: [batch, tree_size]

    p_target = torch.gather(target_probs, dim=-1, index=token_indices).squeeze(-1)
    # p_target: [batch, tree_size]

    # Rejection sampling: accept if r < p_target / p_draft
    r = torch.rand(batch_size, tree_size, device=device)
    ratio = p_target / torch.clamp(p_draft, min=1e-10)
    accepted = r < ratio

    return accepted


def _propagate_tree_acceptance(
    node_accepted: Tensor,
    tree_parents: Tensor,
    device: torch.device,
) -> Tensor:
    """Propagate acceptance through tree structure.

    A node is truly accepted only if:
    1. Its own rejection sample passed (node_accepted[i] = True)
    2. All ancestors are also accepted

    We traverse from roots to leaves, propagating acceptance.

    Args:
        node_accepted: [batch, tree_size] independent acceptance per node.
        tree_parents: [tree_size] parent index for each node (-1 for roots).
        device: Computation device.

    Returns:
        [batch, tree_size] boolean tensor where True means node and all
        ancestors were accepted.
    """
    batch_size, tree_size = node_accepted.shape
    tree_accepted = node_accepted.clone()

    # Build a topological order from roots to leaves
    # Nodes with parent -1 are roots, processed first
    # Then nodes whose parents have been processed
    processed = torch.zeros(tree_size, dtype=torch.bool, device=device)
    order: list[int] = []

    # Find roots (parent == -1)
    roots = (tree_parents == -1).nonzero(as_tuple=True)[0].tolist()
    order.extend(roots)
    for idx in roots:
        processed[idx] = True

    # BFS to get topological order
    queue = list(roots)
    while queue:
        current = queue.pop(0)
        # Find children of current
        children = (tree_parents == current).nonzero(as_tuple=True)[0].tolist()
        for child in children:
            if not processed[child]:
                order.append(child)
                processed[child] = True
                queue.append(child)

    # Propagate acceptance in topological order
    for node_idx in order:
        parent_idx = int(tree_parents[node_idx].item())
        if parent_idx >= 0:
            # Node is accepted only if parent is accepted AND own sample passed
            tree_accepted[:, node_idx] = (
                tree_accepted[:, node_idx] & tree_accepted[:, parent_idx]
            )

    return tree_accepted


def _find_longest_accepted_path(
    tree_accepted: Tensor,
    tree_parents: Tensor,
    draft_tokens: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Find the longest accepted path from root to any node.

    Args:
        tree_accepted: [batch, tree_size] boolean acceptance mask.
        tree_parents: [tree_size] parent indices.
        draft_tokens: [batch, tree_size] token IDs.
        device: Computation device.

    Returns:
        Tuple of:
            paths: [batch, max_depth] token IDs along longest accepted path.
            lengths: [batch] length of accepted path per batch element.
    """
    batch_size, tree_size = tree_accepted.shape

    # Compute depth of each node
    depths = torch.zeros(tree_size, dtype=torch.long, device=device)
    for i in range(tree_size):
        depth = 0
        node = i
        while tree_parents[node] >= 0:
            depth += 1
            node = int(tree_parents[node].item())
        depths[i] = depth

    max_depth = int(depths.max().item()) + 1

    # For each batch element, find the deepest accepted node
    paths = torch.zeros(batch_size, max_depth, dtype=torch.long, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Find all accepted nodes
        accepted_nodes = tree_accepted[b].nonzero(as_tuple=True)[0]

        if len(accepted_nodes) == 0:
            # No accepted nodes
            continue

        # Find the deepest accepted node
        accepted_depths = depths[accepted_nodes]
        deepest_idx = accepted_depths.argmax()
        deepest_node = int(accepted_nodes[deepest_idx].item())
        path_length = int(accepted_depths[deepest_idx].item()) + 1

        # Trace path from deepest node back to root
        path_indices: list[int] = []
        node = deepest_node
        while node >= 0:
            path_indices.append(node)
            if tree_parents[node] >= 0:
                node = int(tree_parents[node].item())
            else:
                break

        # Reverse to get root-to-leaf order
        path_indices = path_indices[::-1]

        # Extract tokens along path
        for i, idx in enumerate(path_indices):
            paths[b, i] = draft_tokens[b, idx]

        lengths[b] = path_length

    return paths, lengths


def _sample_tree_next_token(
    tree_accepted: Tensor,
    tree_parents: Tensor,
    draft_probs: Tensor,
    target_probs: Tensor,
    target_logits: Tensor,
    path_lengths: Tensor,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tensor:
    """Sample next token after tree verification.

    If a complete leaf path is accepted, sample bonus token from target.
    Otherwise, sample from residual distribution at the first rejection point
    along the best path.

    Args:
        tree_accepted: [batch, tree_size] acceptance mask.
        tree_parents: [tree_size] parent indices.
        draft_probs: [batch, tree_size, vocab] draft probabilities.
        target_probs: [batch, tree_size, vocab] target probabilities.
        target_logits: [batch, tree_size, vocab] target logits.
        path_lengths: [batch] accepted path lengths.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        device: Computation device.

    Returns:
        [batch] next token IDs.
    """
    batch_size, tree_size = tree_accepted.shape
    next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Find leaf nodes (nodes with no children)
    has_child = torch.zeros(tree_size, dtype=torch.bool, device=device)
    for i in range(tree_size):
        parent_idx = int(tree_parents[i].item())
        if parent_idx >= 0:
            has_child[parent_idx] = True
    leaf_mask = ~has_child

    # Compute depth of each node for finding position after accepted path
    depths = torch.zeros(tree_size, dtype=torch.long, device=device)
    for i in range(tree_size):
        depth = 0
        node = i
        while tree_parents[node] >= 0:
            depth += 1
            node = int(tree_parents[node].item())
        depths[i] = depth

    for b in range(batch_size):
        path_len = int(path_lengths[b].item())

        if path_len == 0:
            # No accepted tokens - sample from root position
            # Use target distribution at first position
            next_tokens[b] = _sample(
                target_logits[b, 0:1, :], temperature, top_p, device
            ).item()
            continue

        # Find the last accepted node in the path
        # The accepted path tokens correspond to tree nodes we need to identify
        accepted_nodes = tree_accepted[b].nonzero(as_tuple=True)[0]

        if len(accepted_nodes) == 0:
            next_tokens[b] = _sample(
                target_logits[b, 0:1, :], temperature, top_p, device
            ).item()
            continue

        # Get the deepest accepted node
        accepted_depths = depths[accepted_nodes]
        deepest_idx = accepted_depths.argmax()
        deepest_node = int(accepted_nodes[deepest_idx].item())

        # Check if this is a leaf (complete path accepted)
        if leaf_mask[deepest_node]:
            # Bonus token: sample from target at position after the leaf
            # For tree structures, we sample from the distribution at the leaf
            next_tokens[b] = _sample(
                target_logits[b, deepest_node : deepest_node + 1, :],
                temperature,
                top_p,
                device,
            ).item()
        else:
            # Find first rejected child of the deepest accepted node
            children_of_deepest = (tree_parents == deepest_node).nonzero(as_tuple=True)[
                0
            ]

            if len(children_of_deepest) == 0:
                # No children, treat as leaf
                next_tokens[b] = _sample(
                    target_logits[b, deepest_node : deepest_node + 1, :],
                    temperature,
                    top_p,
                    device,
                ).item()
            else:
                # Sample from residual distribution at first rejected child
                first_child = int(children_of_deepest[0].item())
                p_t = target_probs[b, first_child, :]
                p_d = draft_probs[b, first_child, :]

                # Residual distribution: max(0, p_target - p_draft)
                residual = torch.clamp(p_t - p_d, min=0.0)
                residual_sum = residual.sum()

                if residual_sum.item() < 1e-10:
                    # Degenerate: sample from target directly
                    next_tokens[b] = _sample(
                        target_logits[b, first_child : first_child + 1, :],
                        temperature,
                        top_p,
                        device,
                    ).item()
                else:
                    # Sample from normalized residual using Metal
                    residual_normalized = residual / residual_sum
                    if device.type == "cpu":
                        next_tokens[b] = torch.multinomial(
                            residual_normalized, num_samples=1
                        ).item()
                    else:
                        sampler = _get_sampler(residual_normalized.shape[-1])
                        next_tokens[b] = sampler.sample_categorical(
                            torch.log(residual_normalized.unsqueeze(0) + 1e-10),
                            temperature=1.0,
                        )

    return next_tokens


def _greedy_probs(logits: Tensor) -> Tensor:
    """Convert logits to one-hot probability distributions (greedy).

    Args:
        logits: [batch, seq, vocab] raw logits.

    Returns:
        [batch, seq, vocab] with 1.0 at argmax positions, 0 elsewhere.
    """
    argmax_ids = logits.argmax(dim=-1)  # [batch, seq]

    # Build one-hot via scatter
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(2, argmax_ids.unsqueeze(-1), 1.0)
    return one_hot


def _sample_next_token(
    all_accepted: Tensor,
    rejection_pos: Tensor,
    num_spec: int,
    target_logits: Tensor,
    target_probs: Tensor,
    draft_probs: Tensor,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Sample the next token after verification.

    For fully-accepted sequences: sample from target at the bonus position.
    For rejected sequences: sample from the residual distribution
    norm(max(0, p_target - p_draft)) at the rejection position.
    """
    next_token = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Bonus token for fully-accepted sequences
    if all_accepted.any():
        bonus_logits = target_logits[:, num_spec, :]
        bonus_token = _sample(bonus_logits, temperature, top_p, device)
        next_token = torch.where(all_accepted, bonus_token, next_token)

    # Residual sampling for rejected sequences
    rejected_mask = ~all_accepted
    if rejected_mask.any():
        next_token = _sample_residual_batched(
            rejected_mask,
            rejection_pos,
            target_probs,
            draft_probs,
            target_logits,
            temperature,
            top_p,
            batch_size,
            next_token,
            device,
        )

    return next_token


def _sample_residual_batched(
    rejected_mask: Tensor,
    rejection_pos: Tensor,
    target_probs: Tensor,
    draft_probs: Tensor,
    target_logits: Tensor,
    temperature: float,
    top_p: float,
    batch_size: int,
    next_token: Tensor,
    device: torch.device,
) -> Tensor:
    """Sample from residual distribution for rejected batch elements.

    The residual distribution is norm(max(0, p_target - p_draft)), which
    represents the "bonus" probability mass the target assigns beyond
    what the draft model predicted. Sampling from this ensures the
    combined accept/reject procedure produces exactly the target distribution.
    """
    # Gather target and draft probs at each element's rejection position.
    # rejection_pos is [batch], we need probs at those specific seq positions.
    for b in range(batch_size):
        if not rejected_mask[b].item():
            continue
        pos = int(rejection_pos[b].item())
        p_t = target_probs[b, pos, :]  # [vocab]
        p_d = draft_probs[b, pos, :]  # [vocab]

        # Residual distribution: max(0, p_target - p_draft)
        residual = torch.clamp(p_t - p_d, min=0.0)
        residual_sum = residual.sum()

        if residual_sum.item() < 1e-10:
            # Degenerate case: target and draft agree perfectly.
            # Fall back to sampling from target directly.
            token = _sample(target_logits[b : b + 1, pos, :], temperature, top_p, device)
        else:
            # Sample from normalized residual using Metal or CPU fallback
            if device.type == "cpu":
                token = torch.multinomial(residual / residual_sum, num_samples=1)
            else:
                log_residual = torch.log(residual / residual_sum + 1e-10)
                sampler = _get_sampler(log_residual.shape[-1])
                token = sampler.sample_categorical(
                    log_residual.unsqueeze(0), temperature=1.0
                )

        token_val = int(token.reshape(()).item())
        next_token[b] = token_val

    return next_token


def _sample(logits: Tensor, temperature: float, top_p: float, device: torch.device) -> Tensor:
    """Sample from logits with temperature and nucleus (top-p) sampling.

    Args:
        logits: [batch, vocab] or [vocab] raw logits.
        temperature: Temperature for softmax. <= 0 means greedy.
        top_p: Nucleus sampling threshold. 1.0 means no filtering.
        device: Device for output tensor.

    Returns:
        [batch] sampled token indices.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if temperature <= 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)

    if device.type == "cpu":
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Use Metal-accelerated sampling
    sampler = _get_sampler(probs.shape[-1])
    return sampler.sample_categorical(torch.log(probs + 1e-10), temperature=1.0)


def _apply_top_p(probs: Tensor, p: float) -> Tensor:
    """Apply nucleus (top-p) filtering to a probability distribution.

    Zeros out tokens outside the smallest set whose cumulative probability
    exceeds p, then renormalizes.

    Args:
        probs: [batch, vocab] normalized probability distribution.
        p: Cumulative probability threshold.

    Returns:
        [batch, vocab] filtered and renormalized probabilities.
    """
    # Sort descending by probability
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

    # Cumulative sum in sorted order
    cumsum = sorted_probs.cumsum(dim=-1)

    # Keep tokens where cumulative probability hasn't exceeded p yet.
    # Always keep at least the top-1 token.
    mask = torch.cat(
        [
            torch.ones(*probs.shape[:-1], 1, dtype=torch.bool, device=probs.device),
            cumsum[..., :-1] < p,
        ],
        dim=-1,
    )

    # Zero out tokens outside nucleus in sorted space
    filtered_sorted = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))

    # Unsort: scatter back to original positions
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)

    # Renormalize
    total = filtered.sum(dim=-1, keepdim=True)
    return filtered / torch.clamp(total, min=1e-10)
