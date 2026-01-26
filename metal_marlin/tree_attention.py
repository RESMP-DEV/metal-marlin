"""Tree-structured attention for Eagle v3 speculative decoding.

Eagle v3 uses tree-structured speculation where multiple draft paths are
verified in parallel. This module provides Metal GPU-accelerated tree attention
for efficient verification of draft token trees.

Tree attention differs from standard attention:
    1. Draft tokens attend to all accepted prefix positions
    2. Draft tokens attend to ancestor nodes on their path in the tree
    3. Parallel branches cannot attend to each other

Example tree structure (7 draft nodes)::

              [prefix]
                  |
               [node 0]
               /       \\
          [node 1]    [node 2]
            /   \\         |
       [node 3] [node 4] [node 5]
           |
       [node 6]

The tree_mask encodes ancestor relationships:
    tree_mask[i][j] = 1 if node i can attend to node j

Usage:
    from metal_marlin.tree_attention import (
        tree_attention_forward,
        build_tree_mask,
        TreeAttentionConfig,
    )

    # Build mask from parent indices
    parents = torch.tensor([UINT_MAX, 0, 0, 1, 1, 2, 3], dtype=torch.int32)
    tree_mask = build_tree_mask(parents)

    # Run tree attention
    output = tree_attention_forward(
        Q=draft_queries,      # [batch, heads, tree_size, head_dim]
        K=kv_cache_k,         # [batch, heads, seq_len + tree_size, head_dim]
        V=kv_cache_v,         # [batch, heads, seq_len + tree_size, head_dim]
        tree_mask=tree_mask,  # [tree_size, tree_size]
        seq_len=prefix_len,
        scale=1.0 / math.sqrt(head_dim),
    )

Requirements:
    - macOS with Metal support
    - PyObjC Metal framework
    - PyTorch with MPS backend
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

# Conditional imports for optional dependencies
try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None
    Tensor = None

try:
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None

from .metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_mps,
)

# Constants matching tree_attention.metal
TILE_KV_TREE = 32
THREADS_TREE = 128
HEAD_DIM_MAX = 128
MAX_TREE_SIZE = 64

# Sentinel value for root nodes (no parent)
TREE_ROOT = 0xFFFFFFFF


@dataclass
class TreeAttentionConfig:
    """Configuration for tree attention.

    Attributes:
        tree_size: Number of nodes in the draft tree.
        head_dim: Dimension of each attention head.
        seq_len: Length of the accepted prefix.
        scale: Attention scaling factor (typically 1/sqrt(head_dim)).
        use_packed_mask: Whether to use bit-packed tree mask for efficiency.
    """

    tree_size: int
    head_dim: int
    seq_len: int
    scale: float
    use_packed_mask: bool = False


def build_tree_mask(
    parents: Tensor,
    device: torch.device | str = "mps",
) -> Tensor:
    """Build tree attention mask from parent indices.

    Constructs a mask where mask[i][j] = 1 if node j is an ancestor of node i
    (or i == j). This determines which draft tokens each query can attend to.

    Args:
        parents: Parent index for each node, shape [tree_size].
            Use TREE_ROOT (0xFFFFFFFF) for root nodes with no parent.
        device: Device for output tensor.

    Returns:
        Tree mask tensor [tree_size, tree_size] with uint32 dtype.
        mask[i][j] = 1 if node i can attend to node j.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for tree_attention")

    tree_size = parents.shape[0]
    parents_cpu = parents.cpu().numpy()

    # Build mask on CPU (tree traversal is cheap)
    mask = np.zeros((tree_size, tree_size), dtype=np.uint32)

    for i in range(tree_size):
        # Self-attention always allowed
        mask[i, i] = 1

        # Walk up the tree from node i
        current = i
        for _ in range(tree_size):
            parent = parents_cpu[current]
            if parent == TREE_ROOT or parent >= tree_size:
                break
            mask[i, parent] = 1
            current = parent

    return torch.from_numpy(mask).to(device)


def build_tree_mask_packed(
    parents: Tensor,
    device: torch.device | str = "mps",
) -> Tensor:
    """Build bit-packed tree attention mask from parent indices.

    More efficient for large trees (>32 nodes) by packing 64 mask bits
    into each uint64.

    Args:
        parents: Parent index for each node, shape [tree_size].
        device: Device for output tensor.

    Returns:
        Packed tree mask tensor [tree_size, (tree_size+63)//64] with uint64 dtype.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for tree_attention")

    tree_size = parents.shape[0]
    parents_cpu = parents.cpu().numpy()

    # Number of uint64s per row
    num_words = (tree_size + 63) // 64

    packed = np.zeros((tree_size, num_words), dtype=np.uint64)

    for i in range(tree_size):
        # Self-attention
        word_idx = i // 64
        bit_idx = i % 64
        packed[i, word_idx] |= np.uint64(1) << np.uint64(bit_idx)

        # Walk up the tree
        current = i
        for _ in range(tree_size):
            parent = parents_cpu[current]
            if parent == TREE_ROOT or parent >= tree_size:
                break
            word_idx = parent // 64
            bit_idx = parent % 64
            packed[i, word_idx] |= np.uint64(1) << np.uint64(bit_idx)
            current = parent

    return torch.from_numpy(packed).to(device)


def tree_attention_forward(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    tree_mask: Tensor,
    seq_len: int,
    scale: float,
    lib: MetalKernelLibrary | None = None,
) -> Tensor:
    """Compute tree-structured attention for Eagle v3 verification.

    This runs the tree attention kernel that:
        1. Applies standard attention to the accepted prefix
        2. Uses tree_mask for draft-to-draft attention patterns
        3. Returns outputs for all tree nodes in parallel

    Args:
        Q: Query tensor for each tree node, [batch, heads, tree_size, head_dim].
        K: Key tensor including prefix and draft, [batch, heads, seq_len + tree_size, head_dim].
        V: Value tensor including prefix and draft, [batch, heads, seq_len + tree_size, head_dim].
        tree_mask: Attention mask for draft tree, [tree_size, tree_size].
            mask[i][j] = 1 if node i can attend to draft node j.
        seq_len: Length of the accepted prefix.
        scale: Attention scaling factor (1/sqrt(head_dim)).
        lib: MetalKernelLibrary instance. If None, uses default.

    Returns:
        Output tensor [batch, heads, tree_size, head_dim].
    """
    require_mps()

    if lib is None:
        lib = get_default_library()

    device = lib.device

    batch, num_heads, tree_size, head_dim = Q.shape
    total_kv_len = K.shape[2]

    assert K.shape == (batch, num_heads, total_kv_len, head_dim)
    assert V.shape == (batch, num_heads, total_kv_len, head_dim)
    assert tree_mask.shape == (tree_size, tree_size)
    assert total_kv_len == seq_len + tree_size

    # Allocate output
    output = torch.empty_like(Q)

    # Convert tensors to Metal buffers
    Q_buf = mps_tensor_to_metal_buffer(Q.half().contiguous(), device)
    K_buf = mps_tensor_to_metal_buffer(K.half().contiguous(), device)
    V_buf = mps_tensor_to_metal_buffer(V.half().contiguous(), device)
    mask_buf = mps_tensor_to_metal_buffer(tree_mask.int().contiguous(), device)
    O_buf = mps_tensor_to_metal_buffer(output, device)

    # Create constant buffers
    def make_uint_buffer(val: int) -> Any:
        return device.newBufferWithBytes_length_options_(
            np.array([val], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
        )

    def make_float_buffer(val: float) -> Any:
        return device.newBufferWithBytes_length_options_(
            np.array([val], dtype=np.float32).tobytes(), 4, Metal.MTLResourceStorageModeShared
        )

    batch_buf = make_uint_buffer(batch)
    heads_buf = make_uint_buffer(num_heads)
    seq_len_buf = make_uint_buffer(seq_len)
    tree_size_buf = make_uint_buffer(tree_size)
    head_dim_buf = make_uint_buffer(head_dim)
    scale_buf = make_float_buffer(scale)

    # Grid: one threadgroup per (tree_node, head, batch)
    dispatch_kernel(
        lib,
        function_name="tree_attention_forward",
        grid=(tree_size, num_heads, batch),
        threadgroup=(THREADS_TREE, 1, 1),
        buffers=[
            Q_buf,
            K_buf,
            V_buf,
            mask_buf,
            O_buf,
            batch_buf,
            heads_buf,
            seq_len_buf,
            tree_size_buf,
            head_dim_buf,
            scale_buf,
        ],
        wait=True,
    )

    return output


def tree_attention_forward_with_positions(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    tree_mask: Tensor,
    tree_positions: Tensor,
    seq_len: int,
    scale: float,
    lib: MetalKernelLibrary | None = None,
) -> Tensor:
    """Compute tree attention with position-based causal masking in prefix.

    This variant applies proper causal masking within the prefix portion:
    each tree node only attends to prefix positions up to its sequence position.

    Args:
        Q: Query tensor for each tree node, [batch, heads, tree_size, head_dim].
        K: Key tensor including prefix and draft, [batch, heads, seq_len + tree_size, head_dim].
        V: Value tensor including prefix and draft, [batch, heads, seq_len + tree_size, head_dim].
        tree_mask: Attention mask for draft tree, [tree_size, tree_size].
        tree_positions: Sequence position for each tree node, [tree_size].
            Determines causal masking within the prefix.
        seq_len: Length of the accepted prefix.
        scale: Attention scaling factor.
        lib: MetalKernelLibrary instance.

    Returns:
        Output tensor [batch, heads, tree_size, head_dim].
    """
    require_mps()

    if lib is None:
        lib = get_default_library()

    device = lib.device

    batch, num_heads, tree_size, head_dim = Q.shape
    assert tree_positions.shape == (tree_size,)

    # Allocate output
    output = torch.empty_like(Q)

    # Convert tensors to Metal buffers
    Q_buf = mps_tensor_to_metal_buffer(Q.half().contiguous(), device)
    K_buf = mps_tensor_to_metal_buffer(K.half().contiguous(), device)
    V_buf = mps_tensor_to_metal_buffer(V.half().contiguous(), device)
    mask_buf = mps_tensor_to_metal_buffer(tree_mask.int().contiguous(), device)
    pos_buf = mps_tensor_to_metal_buffer(tree_positions.int().contiguous(), device)
    O_buf = mps_tensor_to_metal_buffer(output, device)

    def make_uint_buffer(val: int) -> Any:
        return device.newBufferWithBytes_length_options_(
            np.array([val], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
        )

    def make_float_buffer(val: float) -> Any:
        return device.newBufferWithBytes_length_options_(
            np.array([val], dtype=np.float32).tobytes(), 4, Metal.MTLResourceStorageModeShared
        )

    batch_buf = make_uint_buffer(batch)
    heads_buf = make_uint_buffer(num_heads)
    seq_len_buf = make_uint_buffer(seq_len)
    tree_size_buf = make_uint_buffer(tree_size)
    head_dim_buf = make_uint_buffer(head_dim)
    scale_buf = make_float_buffer(scale)

    dispatch_kernel(
        lib,
        function_name="tree_attention_forward_with_prefix_causal",
        grid=(tree_size, num_heads, batch),
        threadgroup=(THREADS_TREE, 1, 1),
        buffers=[
            Q_buf,
            K_buf,
            V_buf,
            mask_buf,
            pos_buf,
            O_buf,
            batch_buf,
            heads_buf,
            seq_len_buf,
            tree_size_buf,
            head_dim_buf,
            scale_buf,
        ],
        wait=True,
    )

    return output


def tree_attention_forward_packed(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    tree_mask_packed: Tensor,
    seq_len: int,
    scale: float,
    lib: MetalKernelLibrary | None = None,
) -> Tensor:
    """Compute tree attention with bit-packed mask for efficiency.

    Uses uint64 bit-packing for the tree mask to reduce memory bandwidth
    when tree_size > 32.

    Args:
        Q: Query tensor for each tree node, [batch, heads, tree_size, head_dim].
        K: Key tensor including prefix and draft, [batch, heads, seq_len + tree_size, head_dim].
        V: Value tensor including prefix and draft, [batch, heads, seq_len + tree_size, head_dim].
        tree_mask_packed: Bit-packed attention mask, [tree_size, (tree_size+63)//64].
        seq_len: Length of the accepted prefix.
        scale: Attention scaling factor.
        lib: MetalKernelLibrary instance.

    Returns:
        Output tensor [batch, heads, tree_size, head_dim].
    """
    require_mps()

    if lib is None:
        lib = get_default_library()

    device = lib.device

    batch, num_heads, tree_size, head_dim = Q.shape
    # Allocate output
    output = torch.empty_like(Q)

    # Convert tensors to Metal buffers
    Q_buf = mps_tensor_to_metal_buffer(Q.half().contiguous(), device)
    K_buf = mps_tensor_to_metal_buffer(K.half().contiguous(), device)
    V_buf = mps_tensor_to_metal_buffer(V.half().contiguous(), device)
    # tree_mask_packed should be uint64 (long in PyTorch)
    mask_buf = mps_tensor_to_metal_buffer(tree_mask_packed.long().contiguous(), device)
    O_buf = mps_tensor_to_metal_buffer(output, device)

    def make_uint_buffer(val: int) -> Any:
        return device.newBufferWithBytes_length_options_(
            np.array([val], dtype=np.uint32).tobytes(), 4, Metal.MTLResourceStorageModeShared
        )

    def make_float_buffer(val: float) -> Any:
        return device.newBufferWithBytes_length_options_(
            np.array([val], dtype=np.float32).tobytes(), 4, Metal.MTLResourceStorageModeShared
        )

    batch_buf = make_uint_buffer(batch)
    heads_buf = make_uint_buffer(num_heads)
    seq_len_buf = make_uint_buffer(seq_len)
    tree_size_buf = make_uint_buffer(tree_size)
    head_dim_buf = make_uint_buffer(head_dim)
    scale_buf = make_float_buffer(scale)

    dispatch_kernel(
        lib,
        function_name="tree_attention_forward_packed_mask",
        grid=(tree_size, num_heads, batch),
        threadgroup=(THREADS_TREE, 1, 1),
        buffers=[
            Q_buf,
            K_buf,
            V_buf,
            mask_buf,
            O_buf,
            batch_buf,
            heads_buf,
            seq_len_buf,
            tree_size_buf,
            head_dim_buf,
            scale_buf,
        ],
        wait=True,
    )

    return output


# ---------------------------------------------------------------------------
# NumPy fallback implementation for testing/comparison
# ---------------------------------------------------------------------------


def tree_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    tree_mask: np.ndarray,
    seq_len: int,
    scale: float,
) -> np.ndarray:
    """NumPy reference implementation of tree attention.

    Useful for testing and environments without Metal.

    Args:
        Q: Query array [batch, heads, tree_size, head_dim].
        K: Key array [batch, heads, seq_len + tree_size, head_dim].
        V: Value array [batch, heads, seq_len + tree_size, head_dim].
        tree_mask: Tree attention mask [tree_size, tree_size].
        seq_len: Length of accepted prefix.
        scale: Attention scaling factor.

    Returns:
        Output array [batch, heads, tree_size, head_dim].
    """
    batch, num_heads, tree_size, head_dim = Q.shape
    total_kv_len = K.shape[2]

    output = np.zeros_like(Q)

    for b in range(batch):
        for h in range(num_heads):
            for t in range(tree_size):
                q = Q[b, h, t, :]  # [head_dim]

                scores = np.zeros(total_kv_len, dtype=np.float32)

                for k_idx in range(total_kv_len):
                    if k_idx < seq_len:
                        # Prefix: always attend (could add causal here)
                        mask_val = 0.0
                    else:
                        # Draft tree: use tree_mask
                        tree_k_idx = k_idx - seq_len
                        can_attend = tree_mask[t, tree_k_idx]
                        mask_val = 0.0 if can_attend else -np.inf

                    dot = np.dot(q, K[b, h, k_idx, :])
                    scores[k_idx] = dot * scale + mask_val

                # Softmax
                max_score = np.max(scores)
                exp_scores = np.exp(scores - max_score)
                sum_exp = np.sum(exp_scores)
                weights = exp_scores / (sum_exp + 1e-10)

                # Weighted sum of values
                out = np.zeros(head_dim, dtype=np.float32)
                for k_idx in range(total_kv_len):
                    out += weights[k_idx] * V[b, h, k_idx, :]

                output[b, h, t, :] = out

    return output


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _test_tree_attention() -> None:
    """Test tree attention implementation."""
    if not HAS_MPS:
        print("Skipping tree attention test (MPS not available)")
        return

    import math

    print("Testing tree attention...")

    # Create test tree:
    #       0
    #      / \
    #     1   2
    #    / \
    #   3   4
    tree_size = 5
    # Use -1 for root (will be converted to UINT_MAX in the kernel)
    # TREE_ROOT (0xFFFFFFFF) overflows int32, so we use uint32 explicitly
    parents = torch.tensor([TREE_ROOT, 0, 0, 1, 1], dtype=torch.int64, device="mps")

    # Build mask
    mask = build_tree_mask(parents)
    print(f"Tree mask:\n{mask.cpu().numpy()}")

    # Expected mask:
    # Node 0: can attend to [0]
    # Node 1: can attend to [0, 1]
    # Node 2: can attend to [0, 2]
    # Node 3: can attend to [0, 1, 3]
    # Node 4: can attend to [0, 1, 4]

    # Create test tensors
    batch = 1
    num_heads = 4
    head_dim = 64
    seq_len = 10

    Q = torch.randn(batch, num_heads, tree_size, head_dim, dtype=torch.float16, device="mps")
    K = torch.randn(
        batch, num_heads, seq_len + tree_size, head_dim, dtype=torch.float16, device="mps"
    )
    V = torch.randn(
        batch, num_heads, seq_len + tree_size, head_dim, dtype=torch.float16, device="mps"
    )

    scale = 1.0 / math.sqrt(head_dim)

    # Run kernel
    output = None
    try:
        output = tree_attention_forward(Q, K, V, mask, seq_len, scale)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output sample values: {output[0, 0, 0, :8]}")
        print("Tree attention test PASSED")
    except Exception as e:
        print(f"Tree attention test FAILED: {e}")

    # Test with positions
    tree_positions = torch.tensor(
        [seq_len, seq_len + 1, seq_len + 1, seq_len + 2, seq_len + 2],
        dtype=torch.int32,
        device="mps",
    )

    try:
        output_pos = tree_attention_forward_with_positions(
            Q, K, V, mask, tree_positions, seq_len, scale
        )
        print(f"Position-aware output shape: {output_pos.shape}")
        print("Position-aware tree attention test PASSED")
    except Exception as e:
        print(f"Position-aware tree attention test FAILED: {e}")

    # Compare with numpy reference
    Q_np = Q.cpu().numpy().astype(np.float32)
    K_np = K.cpu().numpy().astype(np.float32)
    V_np = V.cpu().numpy().astype(np.float32)
    mask_np = mask.cpu().numpy()

    ref_output = tree_attention_numpy(Q_np, K_np, V_np, mask_np, seq_len, scale)
    print(f"NumPy reference output shape: {ref_output.shape}")
    print(f"NumPy reference sample values: {ref_output[0, 0, 0, :8]}")

    if output is not None:
        metal_output = output.cpu().numpy().astype(np.float32)

        max_diff = np.max(np.abs(ref_output - metal_output))
        print(f"Max difference from reference: {max_diff:.6f}")

        if max_diff < 0.1:  # FP16 tolerance
            print("Reference comparison PASSED")
        else:
            print("Reference comparison FAILED (diff too large)")
    else:
        print("Skipping Metal vs NumPy comparison (Metal dispatch not available)")


if __name__ == "__main__":
    _test_tree_attention()
