
import torch
import torch.nn as nn
from metal_marlin.layers.mmfp4_moe import MMFP4MoE
from metal_marlin.layers.mmfp4_fused_moe import MMFP4FusedMoE

def test_mmfp4_moe_balance_loss():
    print("Testing MMFP4MoE balance_loss...")
    model = MMFP4MoE(
        n_experts=8,
        n_experts_per_tok=2,
        hidden_size=64,
        moe_intermediate_size=128,
        group_size=32
    )
    model.train()  # Enable training mode

    x = torch.randn(2, 10, 64) # Batch 2, Seq 10, Hidden 64
    
    # Forward pass
    output = model(x)
    
    if model.balance_loss is None:
        print("FAILURE: MMFP4MoE.balance_loss is None after forward pass in training mode.")
        exit(1)
    
    if not isinstance(model.balance_loss, torch.Tensor):
        print(f"FAILURE: MMFP4MoE.balance_loss is not a tensor, got {type(model.balance_loss)}")
        exit(1)
        
    if model.balance_loss.item() <= 0:
         print(f"WARNING: MMFP4MoE.balance_loss is <= 0: {model.balance_loss.item()}")

    print(f"SUCCESS: MMFP4MoE.balance_loss computed: {model.balance_loss.item()}")

def test_mmfp4_fused_moe_balance_loss():
    print("
Testing MMFP4FusedMoE balance_loss...")
    model = MMFP4FusedMoE(
        n_experts=8,
        n_experts_per_tok=2,
        hidden_size=64,
        moe_intermediate_size=128,
        group_size=32
    )
    model.train() # Enable training mode

    x = torch.randn(2, 10, 64)
    
    # Forward pass
    output = model(x)
    
    if model.balance_loss is None:
        print("FAILURE: MMFP4FusedMoE.balance_loss is None after forward pass in training mode.")
        exit(1)

    if not isinstance(model.balance_loss, torch.Tensor):
        print(f"FAILURE: MMFP4FusedMoE.balance_loss is not a tensor, got {type(model.balance_loss)}")
        exit(1)
        
    print(f"SUCCESS: MMFP4FusedMoE.balance_loss computed: {model.balance_loss.item()}")

if __name__ == "__main__":
    try:
        test_mmfp4_moe_balance_loss()
        test_mmfp4_fused_moe_balance_loss()
        print("
All tests passed!")
    except Exception as e:
        print(f"
An error occurred: {e}")
        exit(1)
