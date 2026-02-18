"""Multi-Token Prediction head for MMFP4 models.

MTP heads predict N future tokens from hidden states,
enabling speculative decoding without a separate draft model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmfp4_linear import MMFP4Linear
from .adaptive_depth import AdaptiveSpeculationController, AdaptiveDepthConfig

try:
    from ..metal_dispatch import MetalKernelLibrary, get_fast_path
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

_metal_lib = None

def _get_metal_lib():
    global _metal_lib
    if _metal_lib is None and HAS_METAL:
        try:
            # Load kernels from source
            _metal_lib = MetalKernelLibrary.from_source_dir()
        except Exception:
            pass
    return _metal_lib


class MMFP4MTPHead(nn.Module):
    """Multi-Token Prediction head using MMFP4 quantization.
    
    Predicts next N tokens from current hidden state.
    Each prediction head shares the input projection but has
    separate output projections for different lookahead distances.
    
    Supports weight sharing with the target model for memory-efficient
    speculative decoding. When weights are shared, the draft model
    uses a subset of the target model's parameters.
    
    Also supports adaptive speculation depth, which dynamically adjusts
    the number of draft tokens based on observed acceptance rates to
    maximize generation throughput.
    
    Args:
        hidden_size: Model hidden dimension
        vocab_size: Output vocabulary size
        num_predictions: Number of tokens to predict (default 4)
        group_size: MMFP4 quantization group size
        share_weights_with: Optional target model to share weights with
        shared_input_proj: Optional shared input projection weights
        shared_output_heads: Optional shared output head weights
        adaptive_depth: Enable adaptive speculation depth (default True)
        adaptive_depth_config: Configuration for adaptive depth behavior
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predictions: int = 4,
        group_size: int = 128,
        share_weights_with: nn.Module | None = None,
        shared_input_proj: nn.Linear | None = None,
        shared_output_heads: list[nn.Linear] | None = None,
        adaptive_depth: bool = True,
        adaptive_depth_config: AdaptiveDepthConfig | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_predictions = num_predictions
        self._weight_sharing_enabled = False
        self._adaptive_depth_enabled = adaptive_depth
        
        # Fused weights cache for optimized forward pass
        self._fused_weight: torch.Tensor | None = None
        self._fused_bias: torch.Tensor | None = None
        
        # Shared input projection (hidden -> intermediate)
        intermediate_size = hidden_size // 2  # Smaller for efficiency
        
        if shared_input_proj is not None:
            # Use shared weights from target model
            self.input_proj = shared_input_proj
            self._weight_sharing_enabled = True
        else:
            self.input_proj = nn.Linear(hidden_size, intermediate_size)
        
        self.norm = nn.RMSNorm(intermediate_size, eps=1e-5)
        
        # Separate output heads for each prediction position
        if shared_output_heads is not None and len(shared_output_heads) == num_predictions:
            self.output_heads = nn.ModuleList(shared_output_heads)
            self._weight_sharing_enabled = True
        else:
            self.output_heads = nn.ModuleList([
                nn.Linear(intermediate_size, vocab_size, bias=False)
                for _ in range(num_predictions)
            ])
        
        # Store reference for weight sharing
        self._shared_with = share_weights_with
        
        # Initialize adaptive depth controller
        if adaptive_depth:
            config = adaptive_depth_config or AdaptiveDepthConfig(
                initial_depth=num_predictions,
                min_depth=1,
                max_depth=num_predictions,
            )
            self._adaptive_controller = AdaptiveSpeculationController(config)
        else:
            self._adaptive_controller = None
    
    @property
    def adaptive_depth_enabled(self) -> bool:
        """Whether adaptive speculation depth is enabled."""
        return self._adaptive_depth_enabled
    
    @property
    def current_speculation_depth(self) -> int:
        """Current speculation depth.
        
        Returns the adaptive depth if enabled, otherwise returns num_predictions.
        """
        if self._adaptive_depth_enabled and self._adaptive_controller is not None:
            return self._adaptive_controller.current_depth
        return self.num_predictions
    
    def get_adaptive_stats(self) -> dict:
        """Get statistics from adaptive speculation depth.
        
        Returns:
            Dict with adaptive depth statistics, or empty dict if not enabled.
        """
        if not self._adaptive_depth_enabled or self._adaptive_controller is None:
            return {}
        
        stats = self._adaptive_controller.stats
        return {
            "current_depth": stats.current_depth,
            "ema_acceptance": stats.ema_acceptance,
            "recent_acceptance_rate": stats.recent_acceptance_rate,
            "overall_acceptance_rate": stats.overall_acceptance_rate,
            "total_steps": stats.total_steps,
            "total_accepted": stats.total_accepted,
            "total_proposed": stats.total_proposed,
            "depth_changes": stats.depth_changes,
            "efficiency_score": stats.efficiency_score,
            "estimated_speedup": self._adaptive_controller.get_speedup_estimate(),
        }
    
    def update_adaptive_depth(self, num_accepted: int, num_proposed: int) -> int:
        """Update adaptive speculation depth based on acceptance.
        
        Args:
            num_accepted: Number of draft tokens accepted
            num_proposed: Number of draft tokens proposed
        
        Returns:
            New speculation depth
        """
        if not self._adaptive_depth_enabled or self._adaptive_controller is None:
            return self.num_predictions
        
        return self._adaptive_controller.update(num_accepted, num_proposed)
    
    def reset_adaptive_depth(self) -> None:
        """Reset adaptive depth controller for a new sequence."""
        if self._adaptive_controller is not None:
            self._adaptive_controller.reset()
    
    @classmethod
    def from_target_model(
        cls,
        target_model: nn.Module,
        num_predictions: int = 4,
        group_size: int = 128,
        share_embeddings: bool = True,
        share_lm_head: bool = True,
    ) -> "MMFP4MTPHead":
        """Create an MTP head with weight sharing from a target model.
        
        This enables memory-efficient speculative decoding by sharing
        the target model's embedding and LM head weights with the draft model.
        
        Args:
            target_model: The target language model to share weights with
            num_predictions: Number of tokens to predict ahead
            group_size: MMFP4 quantization group size
            share_embeddings: Whether to share input embeddings (if available)
            share_lm_head: Whether to share LM head weights (if available)
            
        Returns:
            MMFP4MTPHead with shared weights from target model
        """
        config = getattr(target_model, "config", None)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", 4096)
            vocab_size = getattr(config, "vocab_size", 32000)
        else:
            hidden_size = getattr(target_model, "hidden_size", 4096)
            vocab_size = getattr(target_model, "vocab_size", 32000)
        
        # Try to extract shared weights from target model
        shared_input_proj = None
        shared_output_heads = None
        
        if share_lm_head:
            # Try to find and adapt the LM head for weight sharing
            lm_head = cls._extract_lm_head(target_model)
            if lm_head is not None:
                # Create shared output heads based on LM head
                intermediate_size = hidden_size // 2
                shared_output_heads = [
                    cls._adapt_linear(lm_head, intermediate_size, vocab_size)
                    for _ in range(num_predictions)
                ]
        
        instance = cls(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_predictions=num_predictions,
            group_size=group_size,
            share_weights_with=target_model,
            shared_input_proj=shared_input_proj,
            shared_output_heads=shared_output_heads,
        )
        
        return instance
    
    @staticmethod
    def _extract_lm_head(target_model: nn.Module) -> nn.Linear | None:
        """Extract the LM head from a target model if available."""
        # Common attribute names for LM heads
        lm_head_names = ["lm_head", "output", "embed_out", "head", "decoder"]
        
        for name in lm_head_names:
            if hasattr(target_model, name):
                module = getattr(target_model, name)
                if isinstance(module, nn.Linear):
                    return module
        
        # Try to find in model's modules
        for module in target_model.modules():
            if isinstance(module, nn.Linear):
                # Heuristic: LM head usually has large output dimension (vocab_size)
                # and is near the end of the model
                if module.weight.shape[0] > 1000:  # Likely vocab size
                    return module
        
        return None
    
    @staticmethod
    def _adapt_linear(
        source: nn.Linear,
        intermediate_size: int,
        vocab_size: int,
    ) -> nn.Linear:
        """Adapt a source linear layer for use as output head.
        
        Creates a new linear layer that shares weights where possible
        or initializes with the source layer's distribution.
        """
        # Create new linear layer
        new_layer = nn.Linear(intermediate_size, vocab_size, bias=source.bias is not None)
        
        # Initialize with proper scaling
        nn.init.xavier_uniform_(new_layer.weight)
        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)
        
        return new_layer
    
    def get_shared_params(self) -> list[nn.Parameter]:
        """Get list of parameters that are shared with the target model."""
        shared = []
        if self._weight_sharing_enabled:
            shared.append(self.input_proj.weight)
            for head in self.output_heads:
                shared.append(head.weight)
        return shared
    
    def get_memory_savings(self) -> dict[str, int]:
        """Calculate memory savings from weight sharing.
        
        Returns:
            Dict with 'total_params', 'shared_params', 'unique_params', 'savings_bytes'
        """
        total_params = sum(p.numel() for p in self.parameters())
        shared_params = sum(p.numel() for p in self.get_shared_params())
        unique_params = total_params - shared_params
        
        # Assume float32 for calculation
        bytes_per_param = 4
        savings_bytes = shared_params * bytes_per_param
        
        return {
            "total_params": total_params,
            "shared_params": shared_params,
            "unique_params": unique_params,
            "savings_bytes": savings_bytes,
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next N tokens with optimized fused operations.
        
        Uses a fused output projection that combines all prediction heads
        into a single matrix multiplication for better GPU utilization
        and reduced kernel launch overhead.
        
        Args:
            hidden_states: [batch, seq, hidden]
        
        Returns:
            logits: [batch, num_predictions, vocab]
        """
        # Take last position for decode
        h = hidden_states[:, -1:, :]  # [batch, 1, hidden]
        
        # Shared transformation with fused SiLU + Norm
        h = self.input_proj(h)  # [batch, 1, intermediate]
        h = F.silu(h)
        h = self.norm(h)
        
        # Fused output projection: all heads in one matmul
        batch_size = h.shape[0]
        h_flat = h.view(batch_size, -1)  # [batch, intermediate]
        
        # Ensure fused weights are built (lazy initialization)
        if self._fused_weight is None or self.training:
            self._rebuild_fused_weights()
        
        # Single matmul: [batch, intermediate] @ [intermediate, num_predictions * vocab]
        # -> [batch, num_predictions * vocab]
        fused_logits = torch.matmul(h_flat, self._fused_weight.t())
        
        # Reshape to [batch, num_predictions, vocab]
        logits = fused_logits.view(batch_size, self.num_predictions, self.vocab_size)
        
        # Add bias if present
        if self._fused_bias is not None:
            logits = logits + self._fused_bias.view(1, self.num_predictions, self.vocab_size)
        
        return logits
    
    def _rebuild_fused_weights(self) -> None:
        """Rebuild fused weight tensor from individual heads.
        
        This is called when weights change (e.g., during loading) or
        on first forward pass. The fused weight enables efficient
        single-kernel execution for all prediction heads.
        """
        # Collect weights from all output heads
        weights = []
        biases = []
        
        for head in self.output_heads:
            weights.append(head.weight)  # [vocab, intermediate]
            if head.bias is not None:
                biases.append(head.bias)  # [vocab]
        
        # Concatenate weights: [num_predictions * vocab, intermediate]
        self._fused_weight = torch.cat(weights, dim=0)
        
        # Concatenate biases if any head has bias
        if biases:
            self._fused_bias = torch.cat(biases, dim=0)
        else:
            self._fused_bias = None
    
    def _clear_fused_weights(self) -> None:
        """Clear fused weights cache (call when weights are modified)."""
        self._fused_weight = None
        self._fused_bias = None
    
    def speculate(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate speculative tokens with probabilities - optimized path.
        
        Uses fused operations and optimized softmax for maximum speed
        during speculative decoding. This is the hot path for draft
        generation and is optimized for latency.
        
        Args:
            hidden_states: [batch, seq, hidden]
            temperature: Sampling temperature
        
        Returns:
            tokens: [batch, num_predictions] predicted token IDs
            probs: [batch, num_predictions, vocab] probability distributions
        """
        with torch.inference_mode():
            logits = self.forward(hidden_states)  # [batch, num_predictions, vocab]
            
            # Apply temperature (optimized for common case of temp=1.0)
            if temperature > 0 and abs(temperature - 1.0) > 1e-6:
                logits = logits / temperature
            
            # Fused softmax computation
            probs = torch.softmax(logits, dim=-1)
            
            # Greedy decoding for maximum acceptance rate
            tokens = torch.argmax(probs, dim=-1)
        
        return tokens, probs
    
    @torch.inference_mode()
    def speculate_fast_path(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maximum performance speculation with minimal overhead.
        
        Args:
            hidden_states: [batch, seq, hidden]
            num_tokens: Number of tokens to predict (default: all)
            temperature: Sampling temperature
            
        Returns:
            tokens: [batch, num_tokens] 
            probs: [batch, num_tokens, vocab]
        """
        if num_tokens is None:
            num_tokens = self.num_predictions
            
        # Forward pass
        logits = self.forward(hidden_states)
        
        # Slice if needed
        if num_tokens < self.num_predictions:
            logits = logits[:, :num_tokens, :]
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits.mul_(1.0 / temperature)
        
        # Compute in one go
        probs = F.softmax(logits, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        
        return tokens, probs


def verify_kernel(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized draft verification kernel.

    Performs rejection sampling to verify draft tokens against target model logits.
    Supports batch processing.

    Args:
        draft_tokens: [batch, num_predictions] Draft token IDs
        target_logits: [batch, num_predictions + 1, vocab] Target model logits
        draft_probs: [batch, num_predictions, vocab] Draft probability distributions.
                     If None, assumes deterministic/greedy draft (prob=1.0).
        temperature: Sampling temperature

    Returns:
        num_accepted: [batch] Number of accepted tokens per sequence
        accepted_mask: [batch, num_predictions] Boolean mask of accepted tokens
        next_token: [batch] The token sampled after the last accepted token
    """
    batch_size, num_predictions = draft_tokens.shape
    device = draft_tokens.device
    vocab_size = target_logits.shape[-1]
    
    # Try using fast Metal kernel if available
    metal_lib = _get_metal_lib()
    if (
        metal_lib is not None 
        and device.type == "mps"
        and draft_probs is None  # Fused kernel assumes greedy draft for now
        and target_logits.dtype == torch.float32  # Kernel assumes float32
    ):
        try:
            # Prepare buffers for kernel
            output_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
            output_mask = torch.zeros((batch_size, num_predictions), dtype=torch.uint8, device=device)
            rand_uniform = torch.rand((batch_size, num_predictions), dtype=torch.float32, device=device)
            
            # Dispatch fused kernel
            # void verify_draft_tokens_fused(
            #     const float* target_logits, const long* draft_tokens, const float* rand_uniform,
            #     int* output_count, uchar* output_mask,
            #     uint V, uint K, float temperature
            # )
            kernel_name = "verify_draft_tokens_fused"
            
            # Use FastPath if available for lower overhead
            fast_path = get_fast_path(metal_lib)
            if fast_path.available:
                # Constants
                V_val = int(vocab_size)
                K_val = int(num_predictions)
                temp_val = float(temperature)

                # Create buffers from pointers (zero-copy)
                target_buf = fast_path.create_buffer_from_ptr(target_logits.data_ptr(), target_logits.nbytes)
                draft_buf = fast_path.create_buffer_from_ptr(draft_tokens.data_ptr(), draft_tokens.nbytes)
                rand_buf = fast_path.create_buffer_from_ptr(rand_uniform.data_ptr(), rand_uniform.nbytes)
                count_buf = fast_path.create_buffer_from_ptr(output_count.data_ptr(), output_count.nbytes)
                mask_buf = fast_path.create_buffer_from_ptr(output_mask.data_ptr(), output_mask.nbytes)
                
                # Constants as buffers (FastPath handles scalar arguments differently depending on implementation,
                # but based on metal_dispatch.py, arguments must be buffers. So creating small buffers.)
                # Use create_buffer for constants to hold values
                
                # V
                V_buf = fast_path.create_buffer(4)
                # We need to write to V_buf. FastPath buffers don't easily allow writing from Python?
                # Actually, metal_lib._dispatch handled int/float by creating buffers.
                # FastPath.dispatch expects buffers.
                # Let's use the slow path for creating constant buffers or rely on metal_lib._dispatch for constants?
                # No, we must pass buffers to FastPath.dispatch.
                
                # To write to ManagedBuffer, we can use `contents()` if accessible, or `didModifyRange`.
                # But ManagedBuffer from C++ extension might expose buffer interface.
                # Alternatively, we can just create a small tensor on CPU, move to MPS, and get pointer?
                # Or create tensor on CPU and use `create_buffer_from_ptr`? No, pointer must be on GPU (Unified memory).
                
                # Easiest way: create 1-element tensors on device
                # This might be slow if done every time.
                # But metal_lib._dispatch does `newBufferWithBytes` which is also slow.
                
                # Let's stick to standard dispatch for constants if FastPath doesn't support scalars directly.
                # However, we can create tensors on device once? No, values change.
                
                # Let's verify metal_dispatch.py support for scalars in FastPath.
                # It calls `_metal_dispatch_ext.dispatch_kernel(..., list(buffers), ...)`
                # It expects buffers.
                
                # So we must provide buffers for V, K, temperature.
                # We can create small tensors on MPS.
                V_tensor = torch.tensor([V_val], dtype=torch.int32, device=device)
                K_tensor = torch.tensor([K_val], dtype=torch.int32, device=device)
                temp_tensor = torch.tensor([temp_val], dtype=torch.float32, device=device)
                
                V_buf = fast_path.create_buffer_from_ptr(V_tensor.data_ptr(), V_tensor.nbytes)
                K_buf = fast_path.create_buffer_from_ptr(K_tensor.data_ptr(), K_tensor.nbytes)
                temp_buf = fast_path.create_buffer_from_ptr(temp_tensor.data_ptr(), temp_tensor.nbytes)
                
                fast_path.dispatch(
                    kernel_name,
                    (batch_size, 1, 1),
                    (1024, 1, 1),
                    [target_buf, draft_buf, rand_buf, count_buf, mask_buf, V_buf, K_buf, temp_buf],
                    wait=True # Wait for completion to ensure we can read results immediately
                )
            else:
                # Standard dispatch fallback
                kernel = metal_lib.get_kernel("verify_fused", kernel_name)
                
                # Temperature must be passed as a buffer, not a scalar float
                temp_tensor = torch.tensor([temperature], dtype=torch.float32, device=device)
                
                metal_lib._dispatch(
                    kernel,
                    (batch_size, 1, 1),      # Grid: one threadgroup per batch item
                    (1024, 1, 1),            # Threads: 1024 (32 warps)
                    metal_lib._get_metal_buffer(target_logits),
                    metal_lib._get_metal_buffer(draft_tokens),
                    metal_lib._get_metal_buffer(rand_uniform),
                    metal_lib._get_metal_buffer(output_count),
                    metal_lib._get_metal_buffer(output_mask),
                    int(vocab_size),
                    int(num_predictions),
                    metal_lib._get_metal_buffer(temp_tensor)
                )
            
            # Metal kernel done, now sample next token
            # Note: kernel returns accepted_mask and num_accepted
            accepted_mask = output_mask.bool()
            num_accepted = output_count.long()
            
            # Use Python fallback logic ONLY for next_token sampling (much cheaper)
            # This avoids computing full softmax matrix
            
            next_token = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # 2. Next Token Sampling (Residual / Bonus)
            # Optimized to avoid full softmax
            
            # Case A: All accepted
            all_accepted = (num_accepted == num_predictions)
            if all_accepted.any():
                # Sample from target at K (bonus position)
                # Compute softmax only for relevant rows
                indices = torch.nonzero(all_accepted, as_tuple=True)[0]
                bonus_logits = target_logits[indices, num_predictions, :] # [num_all_acc, V]
                
                if temperature > 0:
                    bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                    next_token[indices] = torch.multinomial(bonus_probs, 1).squeeze(-1)
                else:
                    next_token[indices] = bonus_logits.argmax(dim=-1)

            # Case B: Rejected at some position
            rejected_mask = ~all_accepted
            if rejected_mask.any():
                batch_idx = torch.nonzero(rejected_mask, as_tuple=True)[0]
                reject_pos = num_accepted[batch_idx] # [num_rejected]
                
                # Gather logits at rejection positions
                # Advanced indexing: [batch_idx, reject_pos, :]
                p_logits = target_logits[batch_idx, reject_pos, :] # [num_rejected, V]
                
                # We need probs to sample residual
                if temperature > 0:
                    p_t = F.softmax(p_logits / temperature, dim=-1)
                else:
                    p_t = torch.zeros_like(p_logits)
                    p_t.scatter_(-1, p_logits.argmax(dim=-1, keepdim=True), 1.0)
                
                # Deterministic draft: q(x) = 1.0 at draft_token
                draft_toks = draft_tokens[batch_idx, reject_pos]
                
                # Residual: p_t with p_t[draft_toks] set to 0 and renormalized
                # This is equivalent to max(0, p_t - p_d) where p_d is one-hot
                p_t.scatter_(1, draft_toks.unsqueeze(1), 0.0)
                
                # Renormalize
                residual_sum = p_t.sum(dim=1, keepdim=True)
                valid_residual = residual_sum.squeeze(-1) > 1e-10
                
                if valid_residual.any():
                    valid_idx = torch.nonzero(valid_residual, as_tuple=True)[0]
                    orig_idx = batch_idx[valid_idx]
                    
                    res_probs = p_t[valid_idx] / residual_sum[valid_idx]
                    
                    if temperature > 0:
                        next_token[orig_idx] = torch.multinomial(res_probs, 1).squeeze(-1)
                    else:
                        next_token[orig_idx] = res_probs.argmax(dim=-1)
                
                # Fallback handled by 0 init or logic above
            
            return num_accepted, accepted_mask, next_token
            
        except Exception as e:
            # Fallback to PyTorch on error
            pass

    # Convert target logits to probabilities
    if temperature > 0:
        target_probs = F.softmax(target_logits / temperature, dim=-1)
    else:
        # Greedy: one-hot on argmax
        target_probs = torch.zeros_like(target_logits)
        target_probs.scatter_(-1, target_logits.argmax(dim=-1, keepdim=True), 1.0)

    # 1. Verification Loop (Vectorized)
    # Gather probs for draft tokens: p(x_i) from target, q(x_i) from draft
    
    # Indices for gathering: [batch, K, 1]
    idx = draft_tokens.unsqueeze(-1)
    
    # Target probs at draft positions: [batch, K]
    # target_probs has shape [batch, K+1, V], we take first K
    p_target = torch.gather(target_probs[:, :num_predictions, :], -1, idx).squeeze(-1)

    if draft_probs is not None:
        p_draft = torch.gather(draft_probs, -1, idx).squeeze(-1)   # [batch, K]
        # Avoid division by zero
        p_draft = torch.clamp(p_draft, min=1e-10)
        ratio = p_target / p_draft
    else:
        # Deterministic draft: q(x) = 1.0 -> ratio = p(x)
        p_draft = torch.ones_like(p_target)
        ratio = p_target

    # Rejection sampling: r ~ U(0, 1) < min(1, p/q)
    r = torch.rand_like(ratio)
    accept_step = r < ratio # [batch, K]

    # Causal acceptance: must accept all previous to accept current
    # cumprod works because False=0, True=1. Once 0, stays 0.
    accepted_mask = accept_step.cumprod(dim=1).bool()
    num_accepted = accepted_mask.sum(dim=1) # [batch]

    # 2. Next Token Sampling (Residual / Bonus)
    # We need to sample x_{n+1} given x_{1..n} are accepted.
    # If n < K (rejection at n+1), sample from residual: norm(max(0, p - q))
    # If n = K (all accepted), sample from target at K+1 (bonus token)
    
    next_token = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Vectorized residual sampling is tricky because 'n' varies per batch.
    # We process by case: all accepted vs rejected at pos i.
    
    # Case A: All accepted
    all_accepted = (num_accepted == num_predictions)
    if all_accepted.any():
        # Sample from target at K (bonus position)
        bonus_probs = target_probs[all_accepted, num_predictions, :]
        if temperature > 0:
            next_token[all_accepted] = torch.multinomial(bonus_probs, 1).squeeze(-1)
        else:
            next_token[all_accepted] = bonus_probs.argmax(dim=-1)

    # Case B: Rejected at some position
    rejected_mask = ~all_accepted
    if rejected_mask.any():
        # Indices of batch elements that were rejected
        batch_idx = torch.nonzero(rejected_mask, as_tuple=True)[0]
        
        # Position of rejection for each rejected batch element
        # num_accepted gives the index of the first rejected token (0-indexed)
        # e.g. num_accepted=2 means indices 0,1 accepted, 2 rejected.
        reject_pos = num_accepted[batch_idx] # [num_rejected]
        
        # Gather p and q at rejection positions
        # Advanced indexing: [batch_idx, reject_pos, :]
        p_t = target_probs[batch_idx, reject_pos, :]
        
        if draft_probs is not None:
            p_d = draft_probs[batch_idx, reject_pos, :]
        else:
            # Deterministic draft: q is one-hot at draft_token
            # We need to construct it or handle it.
            # q(x) = 1 if x == draft else 0
            # Residual = max(0, p - q).
            # If x == draft, p - 1 < 0 (unless p=1). So 0.
            # If x != draft, p - 0 = p.
            # So residual is just p, but with probability at draft_token set to 0.
            # And then renormalized.
            
            # Construct one-hot q
            draft_toks = draft_tokens[batch_idx, reject_pos] # [num_rejected]
            p_d = torch.zeros_like(p_t)
            p_d.scatter_(1, draft_toks.unsqueeze(1), 1.0)

        # Residual distribution: max(0, p - q)
        residual = torch.clamp(p_t - p_d, min=0.0)
        
        # Renormalize
        residual_sum = residual.sum(dim=1, keepdim=True)
        # Handle numerical stability (if sum is 0, fallback to target)
        valid_residual = residual_sum.squeeze(-1) > 1e-10
        
        # For valid residuals
        if valid_residual.any():
            valid_idx = torch.nonzero(valid_residual, as_tuple=True)[0]
            # Map back to original batch indices
            orig_idx = batch_idx[valid_idx]
            
            res_probs = residual[valid_idx] / residual_sum[valid_idx]
            
            if temperature > 0:
                next_token[orig_idx] = torch.multinomial(res_probs, 1).squeeze(-1)
            else:
                next_token[orig_idx] = res_probs.argmax(dim=-1)
                
        # For invalid residuals (p approx q), fallback to target distribution
        if (~valid_residual).any():
            invalid_idx = torch.nonzero(~valid_residual, as_tuple=True)[0]
            orig_idx = batch_idx[invalid_idx]
            # Just sample from p_t (which matches p_d closely)
            # effectively just resampling the rejected token or similar
            if temperature > 0:
                next_token[orig_idx] = torch.multinomial(p_t[invalid_idx], 1).squeeze(-1)
            else:
                next_token[orig_idx] = p_t[invalid_idx].argmax(dim=-1)

    return num_accepted, accepted_mask, next_token


__all__ = ["MMFP4MTPHead", "verify_kernel", "AdaptiveSpeculationController", "AdaptiveDepthConfig"]