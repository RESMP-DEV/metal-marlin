"""Load-balancing aware dispatch for mixed-precision MoE.

This module implements dispatch strategies that account for variable compute
costs per expert (e.g., due to mixed bit-widths 2/3/4/8-bit). It groups
and schedules expert execution to maximize throughput and minimize
pipeline bubbles.

Key Features:
1. LoadBalancedDispatcher class:
   - Track per-expert compute cost (2-bit < 3-bit < 4-bit < 8-bit)
   - Dynamically batch tokens to maximize throughput
   - Reorder tokens to coalesce same-bit-width experts

2. Batching strategies:
   - Bit-width binning: Process all 2-bit experts together
   - Cost-weighted scheduling: More tokens for cheaper experts
   - Work-stealing between expert compute queues

3. Integration with C++ dispatch:
   - Pass batched groups to batch_dispatch_gemm
   - Minimize command buffer switches
"""

from __future__ import annotations

import collections
import dataclasses
import heapq
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from metal_marlin.moe.moe_dispatch_metal import ParallelExpertExecutor


@dataclasses.dataclass
class ExpertGroup:
    """A group of experts with the same compute cost (bit-width)."""
    bit_width: int
    expert_indices: list[int]
    cost_weight: float = 1.0
    estimated_compute_ms: float = 0.0  # Estimated compute time in milliseconds


@dataclasses.dataclass
class DispatchBatch:
    """A prepared batch of tokens for a set of experts."""
    expert_indices: list[int]
    token_indices: torch.Tensor
    expert_offsets: torch.Tensor
    sorted_tokens: torch.Tensor
    bit_width: int = 4
    compute_cost: float = 1.0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LoadBalanceMetrics:
    """Metrics for load balancing performance analysis."""
    total_tokens: int = 0
    total_experts: int = 0
    bit_width_groups: dict[int, int] = dataclasses.field(default_factory=dict)
    expert_utilization: dict[int, int] = dataclasses.field(default_factory=dict)
    batch_sizes: list[int] = dataclasses.field(default_factory=list)
    scheduling_latency_ms: float = 0.0
    estimated_speedup: float = 1.0


class ExpertComputeQueue:
    """Thread-safe queue for expert compute tasks with work-stealing support."""
    
    def __init__(self, worker_id: int, max_size: int = 1000):
        self.worker_id = worker_id
        self.queue = collections.deque()
        self.lock = threading.RLock()
        self.total_load = 0.0
        self.bit_width_cache: dict[int, list[int]] = {}
        
    def push(self, batch: DispatchBatch) -> None:
        """Push a batch to the queue."""
        with self.lock:
            self.queue.append(batch)
            self.total_load += batch.compute_cost
            
            # Cache by bit-width for efficient stealing
            bit_width = batch.bit_width
            if bit_width not in self.bit_width_cache:
                self.bit_width_cache[bit_width] = []
            self.bit_width_cache[bit_width].append(len(self.queue) - 1)
    
    def pop(self) -> Optional[DispatchBatch]:
        """Pop a batch from the queue (LIFO for cache locality)."""
        with self.lock:
            if not self.queue:
                return None
            batch = self.queue.pop()
            self.total_load -= batch.compute_cost
            
            # Update bit-width cache
            bit_width = batch.bit_width
            if bit_width in self.bit_width_cache and self.bit_width_cache[bit_width]:
                self.bit_width_cache[bit_width].pop()
                if not self.bit_width_cache[bit_width]:
                    del self.bit_width_cache[bit_width]
            
            return batch
    
    def steal(self, target_bit_width: Optional[int] = None) -> Optional[DispatchBatch]:
        """Steal a batch from this queue (FIFO for work distribution).
        
        Args:
            target_bit_width: If specified, steal a batch of this bit-width.
                              If None, steal any batch.
        """
        with self.lock:
            if not self.queue:
                return None
            
            if target_bit_width is not None:
                # Try to steal a batch of specific bit-width
                if target_bit_width in self.bit_width_cache and self.bit_width_cache[target_bit_width]:
                    idx = self.bit_width_cache[target_bit_width][0]
                    if 0 <= idx < len(self.queue):
                        batch = self.queue[idx]
                        # Remove from queue
                        self.queue.remove(batch)
                        self.total_load -= batch.compute_cost
                        # Update cache
                        self.bit_width_cache[target_bit_width].remove(idx)
                        if not self.bit_width_cache[target_bit_width]:
                            del self.bit_width_cache[target_bit_width]
                        return batch
            
            # Steal from front (oldest task)
            batch = self.queue.popleft()
            self.total_load -= batch.compute_cost
            
            # Update bit-width cache
            bit_width = batch.bit_width
            if bit_width in self.bit_width_cache and self.bit_width_cache[bit_width]:
                # Shift indices since we removed from front
                self.bit_width_cache[bit_width] = [i-1 for i in self.bit_width_cache[bit_width] if i > 0]
                if not self.bit_width_cache[bit_width]:
                    del self.bit_width_cache[bit_width]
            
            return batch
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self.lock:
            return len(self.queue) == 0
    
    def size(self) -> int:
        """Get queue size."""
        with self.lock:
            return len(self.queue)


class LoadBalancedDispatcher(nn.Module):
    """Dispatcher that balances compute load across mixed-precision experts.

    Tracks per-expert compute cost (2-bit < 3-bit < 4-bit < 8-bit) and
    dynamically batches tokens to maximize throughput.
    
    Key optimizations:
    1. Bit-width binning: Group experts by bit-width to minimize kernel switches
    2. Cost-weighted scheduling: Assign more tokens to cheaper experts
    3. Work-stealing: Balance load across parallel compute queues
    4. Dynamic batching: Adjust batch sizes based on expert cost
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        expert_costs: dict[int, float] | None = None,
        default_bit_width: int = 4,
        available_workers: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.default_bit_width = default_bit_width
        self.available_workers = available_workers
        
        # Default cost model: bit-width determines relative cost
        # 2-bit: 0.5x, 3-bit: 0.75x, 4-bit: 1.0x, 8-bit: 2.0x
        self.bit_width_cost_map = {
            2: 0.5,
            3: 0.75,
            4: 1.0,
            8: 2.0,
        }
        
        # Initialize expert costs if not provided
        if expert_costs is None:
            expert_costs = {i: 1.0 for i in range(num_experts)}
        self.expert_costs = expert_costs
        
        # Map from cost to bit-width (inverse of above for estimation)
        self.cost_to_bit_width = {v: k for k, v in self.bit_width_cost_map.items()}
        
        # Metrics tracking - MUST be initialized before _rebuild_expert_groups
        self.metrics = LoadBalanceMetrics()
        self.metrics.total_experts = num_experts
        
        # Internal state for scheduling
        self.expert_groups: list[ExpertGroup] = []
        self.compute_queues: list[ExpertComputeQueue] = []
        self._rebuild_expert_groups()
        self._init_compute_queues(available_workers)

    def _rebuild_expert_groups(self):
        """Group experts by their cost for efficient binning."""
        by_cost = collections.defaultdict(list)
        for idx, cost in self.expert_costs.items():
            by_cost[cost].append(idx)
        
        self.expert_groups = []
        # Sort by cost ascending (2-bit first)
        for cost, indices in sorted(by_cost.items()):
            # Map cost to approximate bit-width
            bit_width = self.default_bit_width
            for bw, bw_cost in self.bit_width_cost_map.items():
                if abs(cost - bw_cost) < 0.1:
                    bit_width = bw
                    break
            
            # Estimate compute time based on bit-width
            estimated_ms = bit_width * 0.1  # Simplified model
            
            self.expert_groups.append(ExpertGroup(
                bit_width=bit_width,
                expert_indices=indices,
                cost_weight=cost,
                estimated_compute_ms=estimated_ms
            ))
        
        # Update metrics
        self.metrics.bit_width_groups = {
            group.bit_width: len(group.expert_indices)
            for group in self.expert_groups
        }

    def _init_compute_queues(self, num_workers: int):
        """Initialize compute queues for work-stealing."""
        self.compute_queues = [
            ExpertComputeQueue(worker_id=i, max_size=1000)
            for i in range(num_workers)
        ]

    def update_expert_costs(self, new_costs: dict[int, float]):
        """Update per-expert compute costs and rebuild groups."""
        self.expert_costs.update(new_costs)
        self._rebuild_expert_groups()

    def _estimate_bit_width_from_cost(self, cost: float) -> int:
        """Estimate bit-width from compute cost."""
        closest_bit_width = self.default_bit_width
        min_diff = float('inf')
        
        for bit_width, bit_cost in self.bit_width_cost_map.items():
            diff = abs(cost - bit_cost)
            if diff < min_diff:
                min_diff = diff
                closest_bit_width = bit_width
        
        return closest_bit_width

    def schedule_batches(
        self,
        expert_counts: torch.Tensor,
        token_to_expert_map: torch.Tensor,
        sorted_tokens: torch.Tensor,
        expert_offsets: torch.Tensor,
        available_workers: Optional[int] = None
    ) -> list[DispatchBatch]:
        """Schedule expert execution using cost-weighted logic and binning.
        
        Args:
            expert_counts: Tensor of shape [num_experts] containing token counts.
            token_to_expert_map: Tensor mapping tokens to experts.
            sorted_tokens: Tokens sorted by expert assignment.
            expert_offsets: Offsets for expert boundaries in sorted tokens.
            available_workers: Number of parallel execution streams (virtual workers).
            
        Returns:
            List of DispatchBatch objects ready for execution.
        """
        start_time = time.time()
        
        if available_workers is None:
            available_workers = self.available_workers
        
        # 1. Bit-width binning: Process groups sorted by bit-width
        # This minimizes command buffer switches between different kernels (2bit vs 4bit kernels)
        
        # Group experts by bit-width
        bit_width_to_experts = collections.defaultdict(list)
        for group in self.expert_groups:
            for expert_idx in group.expert_indices:
                count = expert_counts[expert_idx].item()
                if count > 0:
                    bit_width_to_experts[group.bit_width].append((expert_idx, group, count))
        
        # Sort bit-widths by compute cost (ascending)
        sorted_bit_widths = sorted(bit_width_to_experts.keys(), 
                                  key=lambda bw: self.bit_width_cost_map.get(bw, 1.0))
        
        # Priority queue for work-stealing (min-heap of current load per worker)
        # (current_load, worker_id)
        worker_loads = [(0.0, i) for i in range(available_workers)]
        heapq.heapify(worker_loads)
        
        # Assign experts to workers to balance load (Longest Processing Time First)
        worker_assignments = collections.defaultdict(list)
        
        # Process each bit-width group
        for bit_width in sorted_bit_widths:
            experts_in_group = bit_width_to_experts[bit_width]
            
            # Sort experts within group by total work (count * cost) descending
            experts_in_group.sort(key=lambda x: x[2] * x[1].cost_weight, reverse=True)
            
            for expert_idx, group, count in experts_in_group:
                # "Steal" work: assign to the worker with the least current load
                current_load, worker_id = heapq.heappop(worker_loads)
                
                worker_assignments[worker_id].append((expert_idx, group, count, bit_width))
                
                # Update load with estimated compute time
                work_estimate = count * group.cost_weight
                new_load = current_load + work_estimate
                heapq.heappush(worker_loads, (new_load, worker_id))
        
        # 2. Construct DispatchBatches per worker
        batches = []
        
        for worker_id, assignments in worker_assignments.items():
            if not assignments:
                continue
            
            # Group by bit-width within the worker's queue
            current_bit_width = None
            current_batch_experts = []
            
            # Sort assignments by bit-width for kernel locality
            assignments.sort(key=lambda x: x[3])  # x[3] is bit_width
            
            for expert_idx, group, count, bit_width in assignments:
                if bit_width != current_bit_width and current_batch_experts:
                    # Create batch for current bit-width
                    batch = self._create_batch_for_experts(
                        current_batch_experts,
                        expert_counts,
                        token_to_expert_map,
                        sorted_tokens,
                        expert_offsets,
                        current_bit_width
                    )
                    if batch:
                        batches.append(batch)
                        self.metrics.batch_sizes.append(len(current_batch_experts))
                    current_batch_experts = []
                    current_bit_width = bit_width
                
                if current_bit_width is None:
                    current_bit_width = bit_width
                
                current_batch_experts.append((expert_idx, group, count))
            
            # Create batch for remaining experts
            if current_batch_experts:
                batch = self._create_batch_for_experts(
                    current_batch_experts,
                    expert_counts,
                    token_to_expert_map,
                    sorted_tokens,
                    expert_offsets,
                    current_bit_width
                )
                if batch:
                    batches.append(batch)
                    self.metrics.batch_sizes.append(len(current_batch_experts))
        
        # Update metrics
        self.metrics.scheduling_latency_ms = (time.time() - start_time) * 1000
        self.metrics.total_tokens = expert_counts.sum().item()
        
        # Estimate speedup from load balancing
        if len(batches) > 0:
            avg_batch_size = sum(self.metrics.batch_sizes) / len(self.metrics.batch_sizes)
            self.metrics.estimated_speedup = min(2.0, 1.0 + (avg_batch_size / 10.0))
        
        return batches

    def _create_batch_for_experts(
        self,
        expert_data: list[tuple[int, ExpertGroup, int]],
        expert_counts: torch.Tensor,
        token_to_expert_map: torch.Tensor,
        sorted_tokens: torch.Tensor,
        expert_offsets: torch.Tensor,
        bit_width: int
    ) -> Optional[DispatchBatch]:
        """Create a DispatchBatch for a set of experts with the same bit-width."""
        if not expert_data:
            return None
        
        expert_indices = [expert_idx for expert_idx, _, _ in expert_data]
        
        # Collect all token indices for these experts
        token_indices_list = []
        for expert_idx, _, _ in expert_data:
            # Find tokens assigned to this expert
            count = expert_counts[expert_idx].item()
            if count == 0:
                continue
            
            # Get token range from offsets
            start_idx = expert_offsets[expert_idx].item()
            end_idx = expert_offsets[expert_idx + 1].item()
            
            # Get actual token indices
            for i in range(start_idx, end_idx):
                token_indices_list.append(i)
        
        if not token_indices_list:
            return None
        
        # Create tensor of token indices
        token_indices = torch.tensor(token_indices_list, dtype=torch.long, device=sorted_tokens.device)
        
        # Calculate compute cost for this batch
        total_cost = 0.0
        for _, group, count in expert_data:
            total_cost += count * group.cost_weight
        
        return DispatchBatch(
            expert_indices=expert_indices,
            token_indices=token_indices,
            expert_offsets=expert_offsets,
            sorted_tokens=sorted_tokens,
            bit_width=bit_width,
            compute_cost=total_cost,
            metadata={
                "num_tokens": len(token_indices_list),
                "num_experts": len(expert_indices),
                "bit_width": bit_width,
            }
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        experts: Sequence[nn.Module] | None = None,
    ) -> Any:
        """Dispatch tokens to experts using load-balancing strategy.

        Args:
            hidden_states: [batch, hidden_dim]
            router_logits: [batch, num_experts]
            experts: Optional list of expert modules.

        Returns:
            Combined output [batch, hidden_dim]
        """
        # 1. Routing
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # 2. Global Sort & Count
        flat_indices = topk_indices.flatten()
        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts)
        
        # 3. Create token-to-expert map and sort
        # For simplicity, we'll create a basic mapping
        # In a real implementation, this would use efficient scatter/gather
        batch_size = hidden_states.shape[0]
        token_to_expert = topk_indices[:, 0]  # Primary expert for each token
        
        # Sort tokens by expert
        sorted_tokens = hidden_states
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=hidden_states.device)
        
        # Calculate offsets (simplified)
        for i in range(self.num_experts):
            expert_offsets[i + 1] = expert_offsets[i] + expert_counts[i].item()
        
        # 4. Schedule
        batches = self.schedule_batches(
            expert_counts,
            token_to_expert,
            sorted_tokens,
            expert_offsets,
            available_workers=self.available_workers
        )
        
        # 5. Execute (Integration with C++ dispatch)
        if experts and hasattr(experts, "executor"):
            # Assume experts object has an executor attached
            # This follows the pattern: Pass batched groups to batch_dispatch_gemm
            # batch_dispatch_gemm(hidden_states, batches, experts.executor)
            pass
        
        # For now, return zeros as placeholder
        return torch.zeros_like(hidden_states)

    def get_metrics(self) -> LoadBalanceMetrics:
        """Get current load balancing metrics."""
        return self.metrics

    def reset_metrics(self):
        """Reset load balancing metrics."""
        self.metrics = LoadBalanceMetrics()
        self.metrics.total_experts = self.num_experts


def batch_dispatch_gemm(
    hidden_states: torch.Tensor,
    expert_batches: list[DispatchBatch],
    executor: "ParallelExpertExecutor",
) -> torch.Tensor:
    """Execute batched GEMMs for grouped experts.
    
    Integrates with C++ dispatch by passing batched groups.
    Minimizes command buffer switches by processing bit-width groups together.
    
    Args:
        hidden_states: Input tensor [batch, hidden_dim]
        expert_batches: List of DispatchBatch objects
        executor: ParallelExpertExecutor for Metal dispatch
    
    Returns:
        Output tensor [batch, hidden_dim]
    """
    output = torch.zeros_like(hidden_states)
    
    # Group batches by bit-width for minimal kernel switches
    batches_by_bit_width = collections.defaultdict(list)
    for batch in expert_batches:
        batches_by_bit_width[batch.bit_width].append(batch)
    
    # Process each bit-width group
    for bit_width, batches in batches_by_bit_width.items():
        # Set up kernel for this bit-width
        # In real implementation, this would select appropriate Metal kernel
        kernel_name = f"expert_gemm_bpw{bit_width}"
        
        # Process all batches of this bit-width together
        for batch in batches:
            # Extract data from batch
            expert_indices = batch.expert_indices
            token_indices = batch.token_indices
            
            if len(token_indices) == 0:
                continue
            
            # Get input for these tokens
            batch_input = hidden_states[token_indices]
            
            # In real implementation, this would:
            # 1. Prepare Metal buffers
            # 2. Encode kernel dispatch
            # 3. Execute via executor
            # 4. Accumulate results
            
            # Placeholder: accumulate zeros
            output[token_indices] += 0.0
    
    return output


class WorkStealingScheduler:
    """Dynamic work-stealing scheduler for parallel expert execution."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.queues = [ExpertComputeQueue(i) for i in range(num_workers)]
        self.steal_attempts = 0
        self.successful_steals = 0
        self.steal_lock = threading.RLock()
    
    def schedule(self, batches: list[DispatchBatch]) -> None:
        """Schedule batches across worker queues."""
        # Initial round-robin assignment
        for i, batch in enumerate(batches):
            worker_id = i % self.num_workers
            self.queues[worker_id].push(batch)
    
    def steal_work(self, thief_id: int, target_bit_width: Optional[int] = None) -> Optional[DispatchBatch]:
        """Attempt to steal work from another queue."""
        with self.steal_lock:
            self.steal_attempts += 1
            
            # Find victim with most work
            victims = []
            for i, queue in enumerate(self.queues):
                if i != thief_id and not queue.is_empty():
                    victims.append((queue.total_load, i, queue))
            
            if not victims:
                return None
            
            # Choose victim with highest load
            victims.sort(key=lambda x: x[0], reverse=True)
            
            for _, victim_id, victim_queue in victims:
                stolen = victim_queue.steal(target_bit_width)
                if stolen is not None:
                    self.successful_steals += 1
                    return stolen
            
            return None
    
    def get_queue_stats(self) -> dict[int, dict[str, Any]]:
        """Get statistics for all queues."""
        stats = {}
        for i, queue in enumerate(self.queues):
            stats[i] = {
                "size": queue.size(),
                "total_load": queue.total_load,
                "bit_widths": list(queue.bit_width_cache.keys()),
            }
        return stats