"""Enhanced Expert Memory Pool for Mixed-Precision MoE Weights.

Advanced memory pooling for 64 experts with mixed bit-widths:
- Bit-width aware allocation with segregated storage pools
- LRU eviction with priority for active experts
- Async loading from CPU RAM to GPU with pinned memory
- Expert prefetching based on attention patterns and routing history
- Memory defragmentation when pool pressure is high

Key improvements over the base implementation:
1. Smart LRU that avoids evicting locked experts
2. Attention-guided prefetching with prediction models
3. Improved defragmentation heuristics
4. Streamlined async I/O pipeline
5. Better statistics and monitoring
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .._compat import HAS_TORCH, torch
from .prefetch import ExpertPrefetcher, PrefetchConfig, PrefetchStrategy

if TYPE_CHECKING:
    import torch as torch_typing
    TensorType = torch_typing.Tensor
else:
    TensorType = Any

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPoolConfig:
    """Enhanced configuration for expert memory pool."""
    
    # Memory configuration
    pool_size_mb: int = 4096  # Total pool size in MB
    device: str = "mps"
    num_experts: int = 64
    num_layers: int = 32
    expert_dim: int = 4096
    hidden_dim: int = 11008  # MLP intermediate size
    activation_dtype: str = "float16"
    
    # Pool distribution by bit-width (must sum to 1.0)
    pool_distribution: Dict[int, float] = field(default_factory=lambda: {2: 0.25, 4: 0.50, 8: 0.25})
    
    # Eviction and locking
    default_bit_width: int = 4
    lock_grace_period: float = 2.0  # Seconds after unlock before eligible for eviction
    min_pool_threshold_mb: int = 64  # Minimum pool size per bit-width in MB
    
    # Prefetching
    prefetch_config: PrefetchConfig = field(default_factory=PrefetchConfig)
    prefetch_lookahead: int = 3  # Number of future layers to prefetch
    attention_guided_prefetch: bool = True
    
    # Async operations
    max_async_loads: int = 8  # Maximum concurrent async loads
    pin_memory: bool = True  # Use pinned memory for CPU->GPU transfers
    
    # Defragmentation
    enable_defrag: bool = True
    defrag_threshold: float = 0.90  # Utilization threshold for defrag
    defrag_cooldown: float = 5.0  # Minimum seconds between defrags
    
    # Monitoring
    enable_detailed_stats: bool = True
    stats_update_interval: float = 60.0  # Seconds between stats logging


class EnhancedBitWidthPool:
    """Enhanced sub-pool for specific bit-width with smart LRU eviction.
    
    Key improvements:
    - Avoids evicting recently unlocked experts for grace period
    - Tracks access patterns for better prefetching
    - Supports prioritized eviction based on last use time
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
        
        # Enhanced LRU tracking with timestamps and unlock times
        self.access_times: Dict[Tuple[int, int], float] = {}
        self.unlock_times: Dict[Tuple[int, int], float] = {}
        
        # Access frequency for prefetch prediction
        self.access_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Lock tracking (external)
        self.locked_experts: Dict[Tuple[int, int], int] = defaultdict(int)
        
    def allocate(self, layer_idx: int, expert_idx: int, 
                 avoid_evict: Optional[List[Tuple[int, int]]] = None) -> Optional[int]:
        """Allocate slot for expert with enhanced eviction logic.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            avoid_evict: List of (layer, expert) to avoid evicting
            
        Returns:
            Slot index or None if allocation failed
        """
        key = (layer_idx, expert_idx)
        
        # If already allocated, update LRU and return existing slot
        if key in self.used_slots:
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            return self.used_slots[key]
        
        # If free slots available, allocate
        if self.free_slots:
            slot = self.free_slots.pop()
            self.used_slots[key] = slot
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            return slot
        
        # No free slots - need to evict
        evicted_key = self._select_eviction_candidate(avoid_evict)
        if evicted_key is None:
            return None
        
        # Evict and allocate
        evicted_slot = self.used_slots.pop(evicted_key)
        self.free_slots.append(evicted_slot)
        
        # Clear evicted metadata
        for d in [self.access_times, self.access_counts, self.unlock_times]:
            d.pop(evicted_key, None)
        
        # Allocate to new expert
        self.used_slots[key] = evicted_slot
        self.access_times[key] = time.time()
        self.access_counts[key] = 1
        
        return evicted_slot
    
    def _select_eviction_candidate(self, avoid_evict: Optional[List[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
        """Select expert to evict using enhanced heuristics.
        
        Priority order (highest to lowest):
        1. Not locked
        2. Not recently unlocked (grace period)
        3. Least recently accessed
        4. Least frequently accessed (tiebreaker)
        """
        avoid_set = set(avoid_evict) if avoid_evict else set()
        
        candidates = []
        for key in self.used_slots.keys():
            if key in avoid_set:
                continue
            
            # Check if locked
            if self.locked_experts.get(key, 0) > 0:
                continue
            
            # Check grace period
            unlock_time = self.unlock_times.get(key)
            if unlock_time and (time.time() - unlock_time) < 2.0:  # 2-second grace
                continue
            
            # Calculate eviction score (lower = better to evict)
            last_access = self.access_times.get(key, 0)
            access_count = self.access_counts.get(key, 0)
            
            # Score favors older, less frequently accessed experts
            # Time component: older = higher score
            time_score = time.time() - last_access
            
            # Frequency component: lower count = higher score
            freq_score = 1.0 / (access_count + 1)
            
            # Combined score (time weighted more heavily)
            score = time_score * 0.7 + freq_score * 0.3
            
            candidates.append((score, key))
        
        if not candidates:
            return None
        
        # Return candidate with highest eviction score
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def lock_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Lock expert to prevent eviction."""
        key = (layer_idx, expert_idx)
        self.locked_experts[key] += 1
    
    def unlock_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Unlock expert, making it eligible for eviction after grace period."""
        key = (layer_idx, expert_idx)
        if key in self.locked_experts:
            self.locked_experts[key] -= 1
            if self.locked_experts[key] <= 0:
                del self.locked_experts[key]
                self.unlock_times[key] = time.time()
    
    def get_ptr(self, slot_idx: int) -> TensorType:
        """Get tensor slice for a slot."""
        if self.buffer is None:
            raise RuntimeError("Pool buffer not initialized")
        start = slot_idx * self.slot_size_bytes
        end = start + self.slot_size_bytes
        return self.buffer[start:end]
    
    def get_utilization(self) -> float:
        """Get pool utilization (0.0 to 1.0)."""
        if self.max_slots == 0:
            return 0.0
        return len(self.used_slots) / self.max_slots
    
    def resize(self, new_max_slots: int) -> None:
        """Resize pool with data preservation."""
        if new_max_slots == self.max_slots:
            return
        
        if not HAS_TORCH or torch is None:
            self.max_slots = new_max_slots
            self.free_slots = list(range(len(self.used_slots), new_max_slots))
            return
        
        # Allocate new buffer
        new_buffer = torch.zeros(
            new_max_slots * self.slot_size_bytes,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=False
        )
        
        # Compact existing experts into new buffer
        new_used_slots = {}
        current_slot = 0
        
        # Sort by access time (most recently accessed first)
        sorted_keys = sorted(self.used_slots.keys(), 
                           key=lambda k: self.access_times.get(k, 0), 
                           reverse=True)
        
        for key in sorted_keys:
            if current_slot >= new_max_slots:
                # Out of space - need to evict least recently used
                break
            
            old_slot = self.used_slots[key]
            old_ptr = self.get_ptr(old_slot)
            
            # Copy to new location
            new_start = current_slot * self.slot_size_bytes
            new_end = new_start + self.slot_size_bytes
            new_ptr = new_buffer[new_start:new_end]
            new_ptr.copy_(old_ptr)
            
            new_used_slots[key] = current_slot
            current_slot += 1
        
        # Update state
        self.buffer = new_buffer
        self.used_slots = new_used_slots
        self.max_slots = new_max_slots
        self.free_slots = list(range(current_slot, new_max_slots))
        
        logger.debug(f"Resized {self.bit_width}-bit pool to {new_max_slots} slots")


class EnhancedExpertMemoryPool:
    """Enhanced expert memory pool with advanced features."""
    
    def __init__(self, config: EnhancedPoolConfig):
        self.config = config
        self.device = config.device
        
        if not (HAS_TORCH and torch is not None):
            logger.warning("PyTorch not found, pool will be non-functional")
            return
        
        # Calculate expert sizes
        self.params_per_expert = 3 * config.expert_dim * config.hidden_dim
        self.expert_sizes = {
            bits: (self.params_per_expert * bits) // 8
            for bits in config.pool_distribution.keys()
        }
        
        # Initialize enhanced sub-pools
        total_bytes = config.pool_size_mb * 1024 * 1024
        self.pools: Dict[int, EnhancedBitWidthPool] = {}
        
        for bits, fraction in config.pool_distribution.items():
            pool_bytes = int(total_bytes * fraction)
            pool_bytes = max(pool_bytes, config.min_pool_threshold_mb * 1024 * 1024)
            
            slot_size = self.expert_sizes[bits]
            max_slots = max(1, pool_bytes // slot_size)
            
            self.pools[bits] = EnhancedBitWidthPool(bits, slot_size, max_slots, self.device)
            logger.info(f"Initialized {bits}-bit pool with {max_slots} slots ({pool_bytes / 1024**2:.1f} MB)")
        
        # Expert metadata
        self.expert_bit_widths: Dict[Tuple[int, int], int] = defaultdict(
            lambda: config.default_bit_width
        )
        self.expert_loaders: Dict[Tuple[int, int], Callable[[], TensorType]] = {}
        
        # CPU cache with pinned memory
        self.cpu_cache: Dict[Tuple[int, int], TensorType] = {}
        
        # Enhanced prefetcher
        self.prefetcher = ExpertPrefetcher(
            num_experts=config.num_experts,
            num_layers=config.num_layers,
            config=config.prefetch_config,
            load_fn=self._load_and_store_callback
        )
        
        # Async load management
        self.load_queue = asyncio.Queue()
        self.load_events: Dict[Tuple[int, int], asyncio.Event] = {}
        self.pending_loads: Dict[Tuple[int, int], asyncio.Task] = {}
        
        # Layer attention patterns for prefetching
        self.attention_patterns: Dict[int, TensorType] = {}
        
        # Synchronization
        self.lock = threading.RLock()
        
        # CUDA stream for async operations
        if self.device == "cuda" and HAS_TORCH:
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
        
        # Statistics
        self.stats = defaultdict(int)
        self.stats.update({
            "allocations": 0,
            "evictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "async_loads": 0,
            "prefetch_hits": 0,
            "defrag_count": 0,
            "defrag_bytes_moved": 0,
        })
        self.last_defrag_time = 0.0
        self.last_stats_time = time.time()
        
        # Start async loader if in async mode
        if config.max_async_loads > 0:
            self._start_async_loader()
    
    def _start_async_loader(self) -> None:
        """Start background async loader thread."""
        def loader_worker():
            asyncio.run(self._async_load_worker())
        
        self.loader_thread = threading.Thread(target=loader_worker, daemon=True)
        self.loader_thread.start()
    
    async def _async_load_worker(self) -> None:
        """Async worker for loading experts in background."""
        while True:
            try:
                key, cpu_tensor, bit_width, callback = await self.load_queue.get()
                layer_idx, expert_idx = key
                
                # Load to GPU
                gpu_buffer = await self._async_load_to_gpu(layer_idx, expert_idx, cpu_tensor, bit_width)
                
                # Notify completion
                if callback:
                    callback(gpu_buffer)
                
                self.load_queue.task_done()
                
            except Exception as e:
                logger.error(f"Async load failed: {e}")
                self.stats["async_load_errors"] += 1
    
    async def _async_load_to_gpu(self, layer_idx: int, expert_idx: int, 
                                cpu_tensor: TensorType, bit_width: int) -> TensorType:
        """Async load CPU tensor to GPU."""
        # Allocate GPU buffer
        gpu_buffer = self.allocate(layer_idx, expert_idx, bit_width)
        
        if self.stream and self.device == "cuda":
            # Async copy on CUDA stream
            with torch.cuda.stream(self.stream):
                gpu_buffer.copy_(cpu_tensor, nonblocking=True)
            await asyncio.to_thread(torch.cuda.current_stream().synchronize)
        else:
            # Synchronous copy for non-CUDA
            gpu_buffer.copy_(cpu_tensor)
            if self.device == "mps":
                torch.mps.synchronize()
        
        self.stats["async_loads"] += 1
        return gpu_buffer
    
    def register_expert(self, layer_idx: int, expert_idx: int, bit_width: int,
                       loader: Optional[Callable[[], TensorType]] = None) -> None:
        """Register expert with optional loader."""
        with self.lock:
            self.expert_bit_widths[(layer_idx, expert_idx)] = bit_width
            if loader:
                self.expert_loaders[(layer_idx, expert_idx)] = loader
    
    def add_to_cpu_cache(self, layer_idx: int, expert_idx: int, cpu_tensor: TensorType) -> None:
        """Add expert weights to CPU cache with optional pinning."""
        if not HAS_TORCH or torch is None:
            return
        
        with self.lock:
            key = (layer_idx, expert_idx)
            
            # Ensure tensor is on CPU
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.cpu()
            
            # Pin memory for async transfers if enabled
            if self.config.pin_memory and not cpu_tensor.is_pinned():
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except RuntimeError:
                    pass
            
            self.cpu_cache[key] = cpu_tensor
    
    def allocate(self, layer_idx: int, expert_idx: int, bit_width: Optional[int] = None) -> TensorType:
        """Allocate GPU memory for expert with enhanced eviction."""
        if bit_width is None:
            bit_width = self.expert_bit_widths.get((layer_idx, expert_idx), self.config.default_bit_width)
        
        if bit_width not in self.pools:
            raise ValueError(f"Unsupported bit width: {bit_width}")
        
        with self.lock:
            pool = self.pools[bit_width]
            key = (layer_idx, expert_idx)
            
            # Get list of experts to avoid evicting
            avoid_evict = list(self.pools[bit_width].locked_experts.keys())
            
            # Try allocation
            slot = pool.allocate(layer_idx, expert_idx, avoid_evict)
            
            if slot is None:
                # Try defragmentation if enabled
                if self.config.enable_defrag:
                    self._defragment_internal()
                    slot = pool.allocate(layer_idx, expert_idx, avoid_evict)
                
                if slot is None:
                    # Last resort: try other bit-width pools
                    for other_bits, other_pool in self.pools.items():
                        if other_bits == bit_width:
                            continue
                        
                        # Check if expert would fit in this pool
                        if self.expert_sizes[bit_width] <= other_pool.slot_size_bytes:
                            # Temporarily allocate in larger pool
                            temp_slot = other_pool.allocate(layer_idx, expert_idx, avoid_evict)
                            if temp_slot is not None:
                                logger.warning(f"Expert {key} allocated in {other_bits}-bit pool (overflow)")
                                return other_pool.get_ptr(temp_slot)
                    
                    raise MemoryError(f"OOM in {bit_width}-bit pool after all strategies")
            
            self.stats["allocations"] += 1
            return pool.get_ptr(slot)
    
    def load_expert(self, layer_idx: int, expert_idx: int,
                   data_loader: Optional[Callable[[], TensorType]] = None,
                   async_load: bool = False) -> TensorType:
        """Load expert with optional async loading."""
        key = (layer_idx, expert_idx)
        bit_width = self.expert_bit_widths.get(key, self.config.default_bit_width)
        
        with self.lock:
            # Check cache first
            if bit_width in self.pools:
                pool = self.pools[bit_width]
                if key in pool.used_slots:
                    self.stats["cache_hits"] += 1
                    pool.access_times[key] = time.time()
                    pool.access_counts[key] += 1
                    return pool.get_ptr(pool.used_slots[key])
            
            self.stats["cache_misses"] += 1
            
            # Get loader
            if data_loader is None:
                data_loader = self.expert_loaders.get(key)
            
            if data_loader is None:
                # Check CPU cache
                cpu_tensor = self.cpu_cache.get(key)
                if cpu_tensor is not None:
                    if async_load and self.config.max_async_loads > 0:
                        # Async load
                        return self._async_load_with_callback(layer_idx, expert_idx, cpu_tensor, bit_width)
                    else:
                        # Sync load
                        return self._load_from_cpu(layer_idx, expert_idx, cpu_tensor, bit_width)
                
                raise RuntimeError(f"No loader or CPU cache for expert {key}")
            
            # Load from data loader
            cpu_tensor = data_loader()
            
            # Add to CPU cache for future use
            self.add_to_cpu_cache(layer_idx, expert_idx, cpu_tensor)
            
            if async_load and self.config.max_async_loads > 0:
                return self._async_load_with_callback(layer_idx, expert_idx, cpu_tensor, bit_width)
            else:
                return self._load_from_cpu(layer_idx, expert_idx, cpu_tensor, bit_width)
    
    def _async_load_with_callback(self, layer_idx: int, expert_idx: int,
                                 cpu_tensor: TensorType, bit_width: int) -> TensorType:
        """Start async load and return placeholder or wait."""
        key = (layer_idx, expert_idx)
        
        # Create completion event
        event = asyncio.Event()
        self.load_events[key] = event
        
        # Submit to async queue
        async def load_task():
            gpu_buffer = await self._async_load_to_gpu(layer_idx, expert_idx, cpu_tensor, bit_width)
            event.set()
            return gpu_buffer
        
        # Start task
        task = asyncio.create_task(load_task())
        self.pending_loads[key] = task
        
        # For now, wait synchronously (could return future in async context)
        # In production, this would integrate with async/await
        import asyncio
        asyncio.run(event.wait())
        
        # Get result
        return self.pools[bit_width].get_ptr(self.pools[bit_width].used_slots[key])
    
    def _load_from_cpu(self, layer_idx: int, expert_idx: int,
                      cpu_tensor: TensorType, bit_width: int) -> TensorType:
        """Synchronous load from CPU to GPU."""
        # Allocate GPU buffer
        gpu_buffer = self.allocate(layer_idx, expert_idx, bit_width)
        
        # Copy data
        gpu_buffer.copy_(cpu_tensor)
        
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        
        return gpu_buffer
    
    def prefetch_experts(self, layer_idx: int, expert_ids: List[int],
                        attention_pattern: Optional[TensorType] = None) -> None:
        """Prefetch experts for next token with attention guidance."""
        with self.lock:
            # Store attention pattern for future prefetching
            if attention_pattern is not None and self.config.attention_guided_prefetch:
                self.attention_patterns[layer_idx] = attention_pattern
            
            # Record routing for prediction
            self.prefetcher.record_routing(layer_idx, expert_ids)
            
            # Predict next experts
            next_experts = self.prefetcher.predict_next_experts(layer_idx, attention_pattern)
            
            # Also predict for future layers if configured
            if self.config.prefetch_lookahead > 0:
                for lookahead in range(1, self.config.prefetch_lookahead + 1):
                    future_layer = layer_idx + lookahead
                    if future_layer < self.config.num_layers:
                        future_experts = self._predict_future_layer_experts(
                            layer_idx, future_layer, expert_ids, attention_pattern
                        )
                        next_experts.extend(future_experts)
            
            # Trigger async loads
            for expert_id in set(next_experts):  # Deduplicate
                try:
                    # Start async load without blocking
                    asyncio.run_coroutine_threadsafe(
                        self._async_prefetch_expert(layer_idx, expert_id),
                        asyncio.get_event_loop()
                    )
                except Exception as e:
                    logger.warning(f"Failed to prefetch expert {expert_id}: {e}")
    
    async def _async_prefetch_expert(self, layer_idx: int, expert_id: int) -> None:
        """Async prefetch for a single expert."""
        try:
            # Load without waiting (fire-and-forget)
            await self.load_queue.put((
                (layer_idx, expert_id),
                self.cpu_cache.get((layer_idx, expert_id)),
                self.expert_bit_widths.get((layer_idx, expert_id), self.config.default_bit_width),
                None  # No callback needed for prefetch
            ))
        except Exception as e:
            logger.debug(f"Prefetch for expert {expert_id} failed: {e}")
    
    def _predict_future_layer_experts(self, current_layer: int, future_layer: int,
                                     expert_ids: List[int], 
                                     attention_pattern: Optional[TensorType]) -> List[int]:
        """Predict experts for future layers based on current routing."""
        # Simple heuristic: same experts as current layer
        # More sophisticated: use attention patterns and model architecture
        return expert_ids
    
    def _defragment_internal(self) -> None:
        """Enhanced defragmentation with better heuristics."""
        if not self.config.enable_defrag:
            return
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_defrag_time < self.config.defrag_cooldown:
            return
        
        # Check if defrag needed
        utilizations = {}
        pool_bytes = {}
        
        for bits, pool in self.pools.items():
            util = pool.get_utilization()
            utilizations[bits] = util
            pool_bytes[bits] = pool.max_slots * pool.slot_size_bytes
        
        # Find pools that need help (> defrag_threshold)
        high_util_pools = [
            bits for bits, util in utilizations.items()
            if util > self.config.defrag_threshold
        ]
        
        if not high_util_pools:
            return
        
        # Find pools with spare capacity (< 0.5 * defrag_threshold)
        low_util_pools = [
            bits for bits, util in utilizations.items()
            if util < (0.5 * self.config.defrag_threshold)
            and pool_bytes[bits] > self.config.min_pool_threshold_mb * 1024 * 1024
        ]
        
        if not low_util_pools:
            return
        
        # Select donor and receiver
        # Prioritize helping the most desperate pool
        receiver_bits = max(high_util_pools, key=lambda bits: utilizations[bits])
        
        # Find best donor (lowest utilization with meaningful capacity)
        donor_bits = min(low_util_pools, key=lambda bits: utilizations[bits])
        
        if donor_bits == receiver_bits:
            return
        
        donor_pool = self.pools[donor_bits]
        receiver_pool = self.pools[receiver_bits]
        
        # Calculate bytes to move (capped at 25% of donor's free space)
        donor_free_bytes = len(donor_pool.free_slots) * donor_pool.slot_size_bytes
        bytes_to_move = min(
            int(donor_free_bytes * 0.25),
            int(pool_bytes[donor_bits] * 0.10)  # Max 10% of donor pool
        )
        
        if bytes_to_move <= 0:
            return
        
        # Convert to slots
        slots_to_remove = bytes_to_move // donor_pool.slot_size_bytes
        slots_to_add = bytes_to_move // receiver_pool.slot_size_bytes
        
        if slots_to_remove == 0 or slots_to_add == 0:
            return
        
        logger.info(
            f"Defrag: Moving {bytes_to_move/1024**2:.1f} MB "
            f"from {donor_bits}-bit to {receiver_bits}-bit pool "
            f"({slots_to_remove} slots → {slots_to_add} slots)"
        )
        
        # Resize pools
        donor_pool.resize(donor_pool.max_slots - slots_to_remove)
        receiver_pool.resize(receiver_pool.max_slots + slots_to_add)
        
        # Update stats
        self.stats["defrag_count"] += 1
        self.stats["defrag_bytes_moved"] += bytes_to_move
        self.last_defrag_time = current_time
    
    def lock_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Lock expert to prevent eviction."""
        with self.lock:
            key = (layer_idx, expert_idx)
            bit_width = self.expert_bit_widths.get(key, self.config.default_bit_width)
            if bit_width in self.pools:
                self.pools[bit_width].lock_expert(layer_idx, expert_idx)
    
    def unlock_expert(self, layer_idx: int, expert_idx: int) -> None:
        """Unlock expert."""
        with self.lock:
            key = (layer_idx, expert_idx)
            bit_width = self.expert_bit_widths.get(key, self.config.default_bit_width)
            if bit_width in self.pools:
                self.pools[bit_width].unlock_expert(layer_idx, expert_idx)
    
    def get_expert_buffer(self, layer_idx: int, expert_idx: int) -> Optional[TensorType]:
        """Get GPU buffer for expert if loaded."""
        with self.lock:
            key = (layer_idx, expert_idx)
            bit_width = self.expert_bit_widths.get(key, self.config.default_bit_width)
            
            if bit_width in self.pools:
                pool = self.pools[bit_width]
                if key in pool.used_slots:
                    return pool.get_ptr(pool.used_slots[key])
            
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            pool_stats = {}
            total_used_slots = 0
            total_max_slots = 0
            
            for bits, pool in self.pools.items():
                used = len(pool.used_slots)
                total = pool.max_slots
                free = len(pool.free_slots)
                
                pool_stats[f"bits_{bits}"] = {
                    "total_slots": total,
                    "used_slots": used,
                    "free_slots": free,
                    "utilization": used / total if total > 0 else 0,
                    "locked_experts": len(pool.locked_experts),
                }
                
                total_used_slots += used
                total_max_slots += total
            
            # Calculate overall hit rate
            hits = self.stats["cache_hits"]
            misses = self.stats["cache_misses"]
            total_access = hits + misses
            hit_rate = hits / total_access if total_access > 0 else 0
            
            # Prefetch stats
            prefetch_stats = {}
            try:
                prefetch_stats = self.prefetcher.get_stats()
            except Exception:
                pass
            
            combined_stats = {
                "pools": pool_stats,
                "overall": {
                    "total_used_slots": total_used_slots,
                    "total_max_slots": total_max_slots,
                    "overall_utilization": total_used_slots / total_max_slots if total_max_slots > 0 else 0,
                    "cache_hit_rate": hit_rate,
                    "cpu_cache_size": len(self.cpu_cache),
                    "pending_async_loads": len(self.pending_loads),
                },
                "counters": dict(self.stats),
                "prefetcher": prefetch_stats,
            }
            
            # Log periodic stats if enabled
            if self.config.enable_detailed_stats:
                current_time = time.time()
                if current_time - self.last_stats_time > self.config.stats_update_interval:
                    logger.info(f"Pool stats: {combined_stats}")
                    self.last_stats_time = current_time
            
            return combined_stats
    
    def _load_and_store_callback(self, layer_idx: int, expert_idx: int) -> TensorType:
        """Callback for prefetcher."""
        return self.load_expert(layer_idx, expert_idx, async_load=True)