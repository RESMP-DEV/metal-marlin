"""Expert memory pool for mixed-precision MoE weights.

Manages GPU memory for active experts with support for:
1. Mixed bit-widths (2/4/8-bit) via segregated storage
2. Async loading and prefetching
3. LRU eviction and memory defragmentation
4. Expert prefetching based on attention patterns
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .._compat import HAS_TORCH, torch
from .prefetch import ExpertPrefetcher, PrefetchConfig

if TYPE_CHECKING:
    import torch as torch_typing
    TensorType = torch_typing.Tensor
else:
    TensorType = Any

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for expert memory pool."""
    pool_size_mb: int = 4096  # Total pool size in MB
    device: str = "mps"
    num_experts: int = 64
    num_layers: int = 32
    expert_dim: int = 4096
    hidden_dim: int = 11008  # MLP intermediate size
    activation_dtype: str = "float16"
    enable_defrag: bool = True
    default_bit_width: int = 4
    prefetch_config: PrefetchConfig = field(default_factory=PrefetchConfig)
    

class BitWidthPool:
    """Sub-pool for a specific bit-width (2, 4, or 8 bits).
    
    Manages fixed-size slots for experts of a specific precision.
    """
    
    def __init__(self, bit_width: int, slot_size_bytes: int, max_slots: int, device: str):
        self.bit_width = bit_width
        self.slot_size_bytes = slot_size_bytes
        self.max_slots = max_slots
        self.device = device
        
        # Pre-allocated buffer
        self.buffer: Optional[TensorType] = None
        if HAS_TORCH and torch is not None:
            # Allocate as flat uint8 buffer
            self.buffer = torch.zeros(
                max_slots * slot_size_bytes,
                dtype=torch.uint8,
                device=device,
                pin_memory=False
            )
            
        # Slot management
        self.free_slots = list(range(max_slots))
        self.used_slots: Dict[Tuple[int, int], int] = {}  # (layer, expert) -> slot_idx
        
        # LRU tracking for this sub-pool
        self.lru: OrderedDict[Tuple[int, int], float] = OrderedDict()
        
    def allocate(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        """Allocate a slot for an expert. Returns slot index or None if full."""
        key = (layer_idx, expert_idx)
        
        # If already allocated, return existing slot and update LRU
        if key in self.used_slots:
            self.lru.move_to_end(key)
            return self.used_slots[key]
            
        # If no free slots, try to evict
        if not self.free_slots:
            if not self.lru:
                return None
            evicted_key, _ = self.lru.popitem(last=False)
            slot = self.used_slots.pop(evicted_key)
            self.free_slots.append(slot)
            
        # Allocate
        if not self.free_slots:
             return None

        slot = self.free_slots.pop()
        self.used_slots[key] = slot
        self.lru[key] = time.time()
        return slot
        
    def get_ptr(self, slot_idx: int) -> TensorType:
        """Get tensor slice for a slot."""
        if self.buffer is None:
            raise RuntimeError("Pool buffer not initialized")
        start = slot_idx * self.slot_size_bytes
        end = start + self.slot_size_bytes
        return self.buffer[start:end]
    
    def get_slot_offset(self, slot_idx: int) -> int:
        """Get byte offset for a slot."""
        return slot_idx * self.slot_size_bytes

    def free(self, layer_idx: int, expert_idx: int) -> None:
        """Free a specific expert slot."""
        key = (layer_idx, expert_idx)
        if key in self.used_slots:
            slot = self.used_slots.pop(key)
            self.free_slots.append(slot)
            if key in self.lru:
                del self.lru[key]
    
    def get_utilization(self) -> float:
        """Get pool utilization (0.0 to 1.0)."""
        if self.max_slots == 0:
            return 0.0
        return len(self.used_slots) / self.max_slots
    
    def has_capacity_for(self, num_slots: int) -> bool:
        """Check if pool has capacity for N more slots."""
        return len(self.free_slots) >= num_slots
    
    def get_free_slot_count(self) -> int:
        """Get number of free slots."""
        return len(self.free_slots)

    def resize(self, new_max_slots: int) -> None:
        """Resize the pool to a new number of slots.
        
        This involves:
        1. Allocating a new buffer
        2. Compacting active experts into the start of the new buffer
        3. Updating metadata
        """
        if new_max_slots == self.max_slots:
            return
        
        # Check if we can fit current experts
        num_used = len(self.used_slots)
        if new_max_slots < num_used:
             # We must evict LRU items until we fit
             # Or raise error. For safety, let's evict.
             while len(self.used_slots) > new_max_slots:
                 if not self.lru:
                     break
                 evicted_key, _ = self.lru.popitem(last=False)
                 slot = self.used_slots.pop(evicted_key)
                 # We don't need to put it in free_slots since we are rebuilding
        
        # Allocate new buffer
        if HAS_TORCH and torch is not None:
            new_buffer = torch.zeros(
                new_max_slots * self.slot_size_bytes,
                dtype=torch.uint8,
                device=self.device,
                pin_memory=False
            )
            
            # Move active experts to new buffer (compacting them)
            # We assign new slots 0, 1, 2... for active experts
            new_used_slots = {}
            current_slot = 0
            
            # Iterate in LRU order (least recent first) or arbitrary?
            # Order doesn't strictly matter for location, but let's keep it stable
            for key, old_slot in self.used_slots.items():
                # Copy data
                old_ptr = self.get_ptr(old_slot)
                
                # New ptr
                start = current_slot * self.slot_size_bytes
                end = start + self.slot_size_bytes
                new_ptr = new_buffer[start:end]
                
                new_ptr.copy_(old_ptr)
                
                new_used_slots[key] = current_slot
                current_slot += 1
            
            # Update state
            self.buffer = new_buffer
            self.used_slots = new_used_slots
            self.max_slots = new_max_slots
            self.free_slots = list(range(current_slot, new_max_slots))
            
            # Rebuild LRU keys if we evicted? 
            # self.lru keys must match self.used_slots keys
            # We already popped evicted ones from self.lru above
            pass
        else:
             # Non-torch environment (should not happen given checks)
             self.max_slots = new_max_slots
             self.free_slots = list(range(len(self.used_slots), new_max_slots))


class ExpertMemoryPool:
    """Main memory pool class handling mixed precision experts."""

    def __init__(self, config: PoolConfig):
        self.config = config
        self.device = config.device
        
        if not (HAS_TORCH and torch is not None):
            logger.warning("PyTorch not found, ExpertMemoryPool will be non-functional")
            return

        # Calculate expert sizes for different precisions
        # Size = (w1 + w2 + w3) * elements * bits / 8
        # Standard MLP: up_proj, gate_proj, down_proj
        # Elements approx = 3 * expert_dim * hidden_dim
        self.params_per_expert = 3 * config.expert_dim * config.hidden_dim
        
        self.expert_sizes = {
            2: (self.params_per_expert * 2) // 8,
            4: (self.params_per_expert * 4) // 8,
            8: (self.params_per_expert * 8) // 8,
        }
        
        # Initialize sub-pools
        total_bytes = config.pool_size_mb * 1024 * 1024
        
        self.pools: Dict[int, BitWidthPool] = {}
        allocations = {2: 0.25, 4: 0.50, 8: 0.25}
        
        for bits, fraction in allocations.items():
            pool_bytes = int(total_bytes * fraction)
            slot_size = self.expert_sizes[bits]
            max_slots = pool_bytes // slot_size
            if max_slots > 0:
                self.pools[bits] = BitWidthPool(bits, slot_size, max_slots, self.device)
                logger.debug(f"Initialized {bits}-bit pool with {max_slots} slots ({pool_bytes / 1024**2:.1f} MB)")
        
        # Expert bit-width registry (defaults to configured default)
        self.expert_bit_widths: Dict[Tuple[int, int], int] = defaultdict(
            lambda: config.default_bit_width
        )
        
        # Loader Registry
        # Maps (layer, expert) -> loader callable returning CPU tensor
        self.expert_loaders: Dict[Tuple[int, int], Callable[[], TensorType]] = {}
        
        # CPU-side cache for expert weights (pinned memory for async transfer)
        self.cpu_cache: Dict[Tuple[int, int], TensorType] = {}
        
        # Load completion events for synchronization (CUDA only)
        self.load_events: Dict[Tuple[int, int], Any] = {}
        
        # Prefetcher integration
        self.prefetcher = ExpertPrefetcher(
            num_experts=config.num_experts,
            num_layers=config.num_layers,
            config=config.prefetch_config,
            load_fn=self._load_and_store_callback
        )
        
        self.lock = threading.RLock()
        # Create stream for async operations
        if HAS_TORCH and torch is not None:
            if self.device == "cuda":
                self.stream = torch.cuda.Stream()
            elif self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "Stream"):
                try:
                    self.stream = torch.mps.Stream()
                except Exception:
                    self.stream = None
            else:
                self.stream = None
        else:
            self.stream = None
        
        self.event_pool = []
        
        # Track active experts to prevent eviction during usage
        self.active_experts: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Statistics
        self.stats = {
            "allocations": 0,
            "evictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "async_loads": 0,
        }

    def register_expert(self, layer_idx: int, expert_idx: int, bit_width: int, 
                        loader: Optional[Callable[[], TensorType]] = None):
        """Register an expert's bit-width and optional loader."""
        with self.lock:
            self.expert_bit_widths[(layer_idx, expert_idx)] = bit_width
            if loader:
                self.expert_loaders[(layer_idx, expert_idx)] = loader
    
    def add_to_cpu_cache(self, layer_idx: int, expert_idx: int, cpu_tensor: TensorType):
        """Add expert weights to CPU-side cache.
        
        CPU tensor should be pinned memory for efficient async transfer.
        """
        if not HAS_TORCH or torch is None:
            return
        
        with self.lock:
            key = (layer_idx, expert_idx)
            # Ensure tensor is on CPU and pinned
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.cpu()
            
            if not cpu_tensor.is_pinned():
                # Try to pin if not already
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except RuntimeError:
                    # Some tensors can't be pinned
                    pass
            
            self.cpu_cache[key] = cpu_tensor
    
    def _get_cached_cpu_tensor(self, layer_idx: int, expert_idx: int) -> Optional[TensorType]:
        """Get CPU tensor from cache if available."""
        key = (layer_idx, expert_idx)
        return self.cpu_cache.get(key)

    def allocate(self, layer_idx: int, expert_idx: int, bit_width: Optional[int] = None) -> TensorType:
        """Allocate memory for an expert and return the buffer slice.
        
        This handles eviction if necessary.
        """
        if bit_width is None:
            bit_width = self.expert_bit_widths.get((layer_idx, expert_idx), self.config.default_bit_width)
            
        if bit_width not in self.pools:
            raise ValueError(f"Unsupported bit width: {bit_width}")
            
        with self.lock:
            pool = self.pools[bit_width]
            slot = pool.allocate(layer_idx, expert_idx)
            
            if slot is None:
                # Try defragmentation
                self._defragment_internal()
                slot = pool.allocate(layer_idx, expert_idx)
                
                if slot is None:
                    raise MemoryError(
                        f"OOM in {bit_width}-bit pool after defrag. "
                        f"Consider increasing pool size or reducing concurrency."
                    )
            
            self.stats["allocations"] += 1
            return pool.get_ptr(slot)

    def _load_and_store_callback(self, layer_idx: int, expert_idx: int) -> TensorType:
        """Internal callback for prefetcher to load data."""
        return self.load_expert(layer_idx, expert_idx)
    
    def load_expert(self, layer_idx: int, expert_idx: int, 
                   data_loader: Optional[Callable[[], TensorType]] = None) -> TensorType:
        """Load expert into GPU memory.
        
        Returns GPU buffer with expert weights.
        """
        with self.lock:
            # Get bit width
            bit_width = self.expert_bit_widths.get((layer_idx, expert_idx), self.config.default_bit_width)
            
            # Check if already in pool
            if bit_width in self.pools:
                pool = self.pools[bit_width]
                key = (layer_idx, expert_idx)
                if key in pool.used_slots:
                    pool.lru.move_to_end(key)
                    self.stats["cache_hits"] += 1
                    return pool.get_ptr(pool.used_slots[key])
            
            self.stats["cache_misses"] += 1
            
            # Get loader
            if data_loader is None:
                data_loader = self.expert_loaders.get((layer_idx, expert_idx))
                
            if data_loader is None:
                # Check CPU cache
                cpu_tensor = self._get_cached_cpu_tensor(layer_idx, expert_idx)
                if cpu_tensor is not None:
                    return self._load_from_cpu(layer_idx, expert_idx, cpu_tensor, bit_width)
                
                raise RuntimeError(
                    f"No loader or CPU cache for expert ({layer_idx}, {expert_idx})"
                )
            
            # Load from data loader
            cpu_tensor = data_loader()
            return self._load_from_cpu(layer_idx, expert_idx, cpu_tensor, bit_width)
    
    def _load_from_cpu(self, layer_idx: int, expert_idx: int, 
                      cpu_tensor: TensorType, bit_width: int) -> TensorType:
        """Load CPU tensor to GPU pool."""
        if not (HAS_TORCH and torch is not None):
            raise RuntimeError("PyTorch required")
        
        # Allocate GPU buffer
        gpu_buffer = self.allocate(layer_idx, expert_idx, bit_width)
        
        # Async copy using stream
        if self.stream:
            if self.device == "cuda":
                with torch.cuda.stream(self.stream):
                    gpu_buffer.copy_(cpu_tensor, nonblocking=True)
                    # Record event for synchronization
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    self.load_events[(layer_idx, expert_idx)] = event
                self.stats["async_loads"] += 1
            elif self.device == "mps":
                with torch.mps.stream(self.stream):
                    gpu_buffer.copy_(cpu_tensor, nonblocking=True)
                    # MPS doesn't have Events like CUDA in standard torch yet, 
                    # but we can use mps.synchronize() or just rely on the stream.
                    # For now, we'll mark it as async.
                self.stats["async_loads"] += 1
        else:
            # Synchronous copy
            gpu_buffer.copy_(cpu_tensor)
            if self.device == "mps":
                torch.mps.synchronize()
        
        return gpu_buffer
    
    def wait_for_load(self, layer_idx: int, expert_idx: int) -> None:
        """Wait for async load completion."""
        if not (HAS_TORCH and torch is not None):
            return
            
        key = (layer_idx, expert_idx)
        event = self.load_events.get(key)
        
        if event:
            if self.device == "cuda":
                event.synchronize()
            del self.load_events[key]
    
    def prefetch_experts(self, layer_idx: int, expert_ids: List[int], 
                        attention_pattern: Optional[TensorType] = None) -> None:
        """Prefetch predicted experts for next token."""
        with self.lock:
            # Record current routing for prediction
            self.prefetcher.record_routing(layer_idx, expert_ids)
            
            # Predict next experts
            next_experts = self.prefetcher.predict_next_experts(layer_idx, attention_pattern)
            
            # Trigger async loads
            for expert_id in next_experts:
                try:
                    # This will be executed asynchronously by prefetcher
                    self.prefetcher.async_load_experts(layer_idx, [expert_id])
                except Exception as e:
                    logger.warning(f"Failed to prefetch expert {expert_id}: {e}")

    def predict_next_layer_experts(self, layer_idx: int, activations: TensorType, 
                                  predictor: Optional[Callable[[TensorType], List[int]]] = None) -> List[int]:
        """Predict experts for the NEXT layer based on current layer activations.
        
        Args:
            layer_idx: Current layer index.
            activations: Current layer hidden states.
            predictor: Optional callable that takes activations and returns expert IDs.
            
        Returns:
            Predicted expert IDs for layer_idx + 1.
        """
        if layer_idx + 1 >= self.config.num_layers:
            return []
            
        if predictor:
            return predictor(activations)
            
        # Default heuristic: Next layer often uses similar experts to current layer
        # (semantic locality across layers)
        history = self.prefetcher._history.get(layer_idx)
        if history and history.history:
            return history.get_last_experts()
            
        return []
    
    def _defragment_internal(self) -> None:
        """Internal defragmentation logic."""
        if not self.config.enable_defrag:
            return
            
        # Calculate utilization
        utilizations = {}
        pool_bytes = {}
        for bits, pool in self.pools.items():
            utilizations[bits] = pool.get_utilization()
            pool_bytes[bits] = pool.max_slots * pool.slot_size_bytes
        
        logger.debug(f"Pre-defrag utilizations: {utilizations}")
        
        # Identify donor (low utilization) and receiver (high utilization)
        donor_bits = None
        receiver_bits = None
        
        # Simple heuristic:
        # Find most desperate receiver (> 90% full)
        # Find most generous donor (< 20% full)
        
        for bits, util in utilizations.items():
            if util > 0.90:
                # Potential receiver
                if receiver_bits is None or util > utilizations[receiver_bits]:
                    receiver_bits = bits
            
            if util < 0.20:
                # Potential donor
                # Must have meaningful amount of memory to give (e.g. > 10MB)
                if pool_bytes[bits] > 10 * 1024 * 1024:
                    if donor_bits is None or util < utilizations[donor_bits]:
                        donor_bits = bits
        
        if donor_bits is not None and receiver_bits is not None and donor_bits != receiver_bits:
            donor_pool = self.pools[donor_bits]
            receiver_pool = self.pools[receiver_bits]
            
            # How much to move?
            # Take 25% of donor's capacity
            bytes_to_move = int(pool_bytes[donor_bits] * 0.25)
            
            # Ensure we don't reduce donor below its current usage
            used_bytes = len(donor_pool.used_slots) * donor_pool.slot_size_bytes
            max_removable = pool_bytes[donor_bits] - used_bytes - (1024 * 1024) # Leave 1MB headroom
            
            bytes_to_move = min(bytes_to_move, max_removable)
            
            if bytes_to_move > 0:
                logger.info(
                    f"Defrag: Moving {bytes_to_move/1024**2:.1f} MB "
                    f"from {donor_bits}-bit pool to {receiver_bits}-bit pool"
                )
                
                # 1. Resize donor (shrink)
                slots_to_remove = bytes_to_move // donor_pool.slot_size_bytes
                new_donor_slots = donor_pool.max_slots - slots_to_remove
                donor_pool.resize(new_donor_slots)
                
                # 2. Resize receiver (grow)
                slots_to_add = bytes_to_move // receiver_pool.slot_size_bytes
                new_receiver_slots = receiver_pool.max_slots + slots_to_add
                receiver_pool.resize(new_receiver_slots)
                
                self.stats["defrags"] = self.stats.get("defrags", 0) + 1
    
    def defragment(self) -> None:
        """Public defragmentation interface."""
        with self.lock:
            self._defragment_internal()
    
    def lock_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Mark expert as active/in-use to prevent eviction."""
        with self.lock:
            self.active_experts[(layer_idx, expert_idx)] += 1
    
    def unlock_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Release lock on expert."""
        with self.lock:
            key = (layer_idx, expert_idx)
            if key in self.active_experts:
                self.active_experts[key] -= 1
                if self.active_experts[key] == 0:
                    del self.active_experts[key]
    
    def get_expert_buffer(self, layer_idx: int, expert_idx: int) -> Optional[TensorType]:
        """Get GPU buffer for expert if loaded."""
        with self.lock:
            bit_width = self.expert_bit_widths.get((layer_idx, expert_idx), self.config.default_bit_width)
            if bit_width in self.pools:
                pool = self.pools[bit_width]
                key = (layer_idx, expert_idx)
                if key in pool.used_slots:
                    return pool.get_ptr(pool.used_slots[key])
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive pool statistics."""
        with self.lock:
            pool_stats = {}
            for bits, pool in self.pools.items():
                total = pool.max_slots
                used = len(pool.used_slots)
                free = len(pool.free_slots)
                pool_stats[f"bits_{bits}"] = {
                    "total_slots": total,
                    "used_slots": used,
                    "free_slots": free,
                    "utilization": used / total if total > 0 else 0,
                }
            
            combined_stats = {
                "pools": pool_stats,
                **self.stats,
                "cpu_cache_size": len(self.cpu_cache),
                "active_experts": len(self.active_experts),
                "pending_loads": len(self.load_events),
            }
            
            # Add prefetcher stats if available
            try:
                prefetch_stats = self.prefetcher.get_stats()
                combined_stats["prefetcher"] = prefetch_stats
            except Exception:
                pass
            
            return combined_stats