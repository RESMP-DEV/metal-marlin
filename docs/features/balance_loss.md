# Auxiliary Balance Loss for Expert Utilization

## Overview

Added auxiliary loss to balance expert utilization in the Trellis MoE layer. The balance loss penalizes uneven expert loads during training, encouraging more uniform expert usage.

## Implementation

### 1. TrellisMoELayer Changes

**File:** `metal_marlin/trellis/moe.py`

**New Parameter:**
- `aux_loss_weight: float = 0.0` - Weight for auxiliary balance loss (default: disabled)

**New Methods:**
- `_compute_balance_loss(expert_ids)` - Computes coefficient of variation (CV) of expert loads
- `get_aux_loss()` - Returns the auxiliary loss for backpropagation

**New Buffer:**
- `aux_loss` - Stores the computed balance loss

### 2. Balance Loss Computation

The loss uses coefficient of variation (CV = std/mean) as the metric:

```python
# Count tokens per expert
expert_loads = torch.bincount(expert_ids, minlength=num_experts).float()
expert_loads = expert_loads / expert_loads.sum()  # Normalize

# Compute CV loss
mean_load = expert_loads.mean()
std_load = expert_loads.std()
cv_loss = std_load / (mean_load + 1e-10)

# Store weighted loss
self.aux_loss = self.aux_loss_weight * cv_loss
```

### 3. Usage

**Training with balance loss:**

```python
layer = TrellisMoELayer(
    config=config,
    layer_weights=weights,
    router_weight=router_weight,
    layer_idx=0,
    aux_loss_weight=0.01,  # Enable with weight 0.01
)

layer.train()
output = layer(x)
task_loss = compute_task_loss(output, targets)
aux_loss = layer.get_aux_loss()
total_loss = task_loss + aux_loss
total_loss.backward()
```

**Inference without balance loss:**

```python
layer = TrellisMoELayer(
    config=config,
    layer_weights=weights,
    router_weight=router_weight,
    layer_idx=0,
    aux_loss_weight=0.0,  # Disabled
)

layer.eval()
output = layer(x)  # No balance loss overhead
```

## Design Decisions

1. **Optional by default** (`aux_loss_weight=0.0`) - No overhead unless explicitly enabled
2. **Only computed during training** - Check `self.training` flag
3. **Coefficient of variation** - Normalized metric that works across different batch sizes
4. **Buffer storage** - Uses `register_buffer` so loss persists across calls

## Testing

Verified with:
1. Direct logic test showing CV increases with imbalance
2. Integration test confirming loss is computed during training
3. Existing test suite passes (ExpertCache tests)

## Example

See `examples/balance_loss_usage.py` for complete training example.

## Benefits

- **Balanced expert usage** - Prevents experts from being underutilized
- **No overhead when disabled** - Zero-cost abstraction for inference
- **Fine-tuning friendly** - Can be enabled selectively during fine-tuning
- **Normalized metric** - CV works across different batch sizes and expert counts
