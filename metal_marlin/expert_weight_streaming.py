"""
Expert weight streaming optimizer for Apple Silicon Unified Memory.

With 256 experts, not all expert weights need to be in GPU memory at once.
Strategy:
1. Keep router + shared expert in GPU always.
2. Track which experts are "hot" (frequently accessed).
3. Stream cold experts from unified memory on demand.
4. Prefetch predicted experts based on recent routing patterns.
"""

from __future__ import annotations

import collections
import logging
from typing import Any



logger = logging.getLogger(__name__)

class ExpertWeightStreamer:
    """
    Manages on-demand expert weight streaming with LRU + frequency-aware eviction.

    On M4 Max Unified Memory, "streaming" is mostly handled by page faults,
    but we can still improve performance via explicit residency hints and
    prefetching. Router and shared-expert weights are assumed resident by
    the caller; this class handles the remaining non-shared experts.
    """

    def __init__(self, num_experts: int = 256, gpu_cache_size: int = 64):
        """
        Args:
            num_experts: Total number of experts in the model.
            gpu_cache_size: Maximum number of experts to keep GPU-resident.
        """
        logger.debug("initializing %s with num_experts=%s, gpu_cache_size=%s", type(self).__name__, num_experts, gpu_cache_size)
        self.num_experts = num_experts
        self.gpu_cache_size = gpu_cache_size

        # LRU cache of GPU-resident experts
        self.gpu_cache: collections.OrderedDict[int, dict[str, Any]] = collections.OrderedDict()

        # Access-frequency tracking to distinguish hot vs cold experts
        self.access_counts: dict[int, int] = collections.defaultdict(int)

        # Running history of routed expert indices for simple pattern prediction
        self._history: collections.deque[int] = collections.deque(maxlen=1024)

    def prefetch_experts(self, expert_indices: list[int]) -> None:
        """
        Prefetch predicted experts based on recent routing patterns.

        On M4 Max this primarily ensures Metal marks the pages resident
        before the compute encoder begins, avoiding mid-kernel page faults.
        """
        logger.debug("prefetch_experts called with expert_indices=%s", expert_indices)
        for expert_idx in expert_indices:
            if 0 <= expert_idx < self.num_experts and expert_idx not in self.gpu_cache:
                self._load_to_gpu(expert_idx)

    def get_expert_weights(self, expert_idx: int) -> dict[str, Any]:
        """
        Get expert weights, streaming from unified memory on demand if cold.

        Returns a dict representing the expert's state/metadata.
        Weights themselves would be attached by the caller (e.g., Metal buffers).
        """
        logger.debug("get_expert_weights called with expert_idx=%s", expert_idx)
        if not (0 <= expert_idx < self.num_experts):
            raise ValueError(
                f"expert_idx {expert_idx} out of range [0, {self.num_experts})"
            )

        # Update hotness tracking and routing history
        self.access_counts[expert_idx] += 1
        self._history.append(expert_idx)

        if expert_idx in self.gpu_cache:
            # Already resident -- move to MRU position
            self.gpu_cache.move_to_end(expert_idx)
            return self.gpu_cache[expert_idx]

        # Cold miss -- stream from unified memory (page fault) into GPU cache
        self._load_to_gpu(expert_idx)
        return self.gpu_cache[expert_idx]

    def evict_cold(self) -> None:
        """
        Evict cold experts to stay within the GPU memory budget.

        Uses a frequency-aware LRU approach to preserve genuinely hot experts.
        """
        logger.debug("evict_cold called")
        while len(self.gpu_cache) > self.gpu_cache_size:
            # Build list of oldest items
            oldest = list(self.gpu_cache.keys())
            # Consider only the oldest half of the cache (or at least 1)
            candidates = oldest[: max(1, len(oldest) // 2)]

            # Evict the expert in the oldest half with the lowest access count
            coldest = min(candidates, key=lambda idx: self.access_counts[idx])
            del self.gpu_cache[coldest]

            # Decay count so historically hot experts can cool down over time
            if self.access_counts[coldest] > 0:
                self.access_counts[coldest] //= 2

    def predicted_experts(self, top_k: int = 8) -> list[int]:
        """
        Return a ranked list of likely-next experts based on recent frequency.

        Useful for callers that want to trigger background prefetching.
        """
        logger.debug("predicted_experts called with top_k=%s", top_k)
        if not self._history:
            return []

        # Simple frequency-based prediction from recent history
        recent_window = collections.deque(self._history, maxlen=256)
        freq: dict[int, int] = collections.defaultdict(int)
        for idx in recent_window:
            freq[idx] += 1

        # Sort by frequency descending; exclude already-resident experts
        ranked = sorted(
            (idx for idx in freq if idx not in self.gpu_cache),
            key=lambda idx: freq[idx],
            reverse=True,
        )
        return ranked[:top_k]

    def _load_to_gpu(self, expert_idx: int) -> None:
        """
        Internal helper to mark an expert as GPU-resident.

        In a real Metal implementation this would wrap:
          - useResource / makeResident on the backing MTLBuffer
          - setPurgeableState(.nonVolatile) to hint residency
        """
        logger.info("_load_to_gpu called with expert_idx=%s", expert_idx)
        self.gpu_cache[expert_idx] = {
            "expert_idx": expert_idx,
            "status": "resident",
        }
        self.gpu_cache.move_to_end(expert_idx)

        if len(self.gpu_cache) > self.gpu_cache_size:
            self.evict_cold()
