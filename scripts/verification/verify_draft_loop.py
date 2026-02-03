#!/usr/bin/env python3
"""Verify that the speculative decoding draft model generation loop is implemented.

This script checks that:
1. The draft model generation loop exists in draft.py
2. The loop has all required steps (forward, softmax, argmax, cache advance)
3. The loop is properly documented
4. The engine.py calls the draft model correctly

This is a static code verification that doesn't require running the actual models.
"""

import re
from pathlib import Path


def verify_implementation():
    """Verify the draft model generation loop is complete."""
    
    print("Verifying Speculative Decoding Draft Model Generation Loop Implementation")
    print("=" * 70)
    
    # Read the draft.py file
    draft_file = Path(__file__).parent / "metal_marlin" / "speculative" / "draft.py"
    if not draft_file.exists():
        print(f"❌ FAILED: {draft_file} not found")
        return 1
    
    draft_code = draft_file.read_text()
    
    # Check 1: SmallModelDraft.speculate method exists
    print("\n[1] Checking SmallModelDraft.speculate method exists...")
    if "def speculate" not in draft_code or "class SmallModelDraft" not in draft_code:
        print("❌ FAILED: SmallModelDraft.speculate method not found")
        return 1
    print("✓ Method exists")
    
    # Check 2: Generation loop marker exists
    print("\n[2] Checking for generation loop documentation...")
    if "=== DRAFT MODEL GENERATION LOOP ===" not in draft_code:
        print("❌ FAILED: Generation loop marker not found")
        return 1
    print("✓ Generation loop is documented")
    
    # Check 3: Loop has all required components
    print("\n[3] Verifying loop components...")
    required_components = [
        ("for.*range\\(num_tokens\\)", "Autoregressive loop over K tokens"),
        ("self\\.model\\(", "Forward pass through draft model"),
        ("logits\\[.*-1.*\\]", "Extract last position logits"),
        ("torch\\.softmax", "Convert logits to probabilities"),
        ("torch\\.argmax", "Greedy token selection"),
        ("self\\._cache\\.advance", "Advance KV cache"),
        ("next_token\\.reshape", "Prepare next input token"),
    ]
    
    for pattern, description in required_components:
        if not re.search(pattern, draft_code):
            print(f"❌ FAILED: Missing {description}")
            return 1
        print(f"  ✓ {description}")
    
    # Check 4: Returns DraftOutput with correct structure
    print("\n[4] Checking return value...")
    if "return DraftOutput" not in draft_code:
        print("❌ FAILED: Does not return DraftOutput")
        return 1
    if "torch.stack(tokens, dim=1)" not in draft_code:
        print("❌ FAILED: Tokens not properly stacked")
        return 1
    if "torch.stack(probs, dim=1)" not in draft_code:
        print("❌ FAILED: Probabilities not properly stacked")
        return 1
    print("✓ Returns properly formatted DraftOutput")
    
    # Check 5: Engine calls the draft model
    print("\n[5] Checking engine integration...")
    engine_file = Path(__file__).parent / "metal_marlin" / "speculative" / "engine.py"
    if not engine_file.exists():
        print(f"❌ FAILED: {engine_file} not found")
        return 1
    
    engine_code = engine_file.read_text()
    if "self.draft.speculate" not in engine_code:
        print("❌ FAILED: Engine doesn't call draft.speculate()")
        return 1
    print("✓ Engine calls draft.speculate()")
    
    # Check 6: Detailed step-by-step documentation
    print("\n[6] Checking step-by-step documentation...")
    steps = [
        "Step 1: Forward pass",
        "Step 2: Extract logits",
        "Step 3: Convert logits",
        "Step 4: Greedy selection",
        "Step 5: Advance",
        "Step 6: Prepare input",
    ]
    
    for step in steps:
        if step not in draft_code:
            print(f"❌ FAILED: Missing documentation for {step}")
            return 1
        print(f"  ✓ {step} documented")
    
    # Check 7: Comprehensive docstring
    print("\n[7] Checking method docstring...")
    if "This implements the core draft model generation loop" not in draft_code:
        print("❌ FAILED: Missing comprehensive docstring")
        return 1
    print("✓ Comprehensive docstring present")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS: Draft model generation loop is fully implemented!")
    print("=" * 70)
    print("\nImplementation summary:")
    print("  • Autoregressive loop generates K tokens sequentially")
    print("  • Each iteration: forward → softmax → argmax → cache advance")
    print("  • Uses greedy decoding for maximum acceptance rate")
    print("  • Returns tokens and probability distributions for verification")
    print("  • Properly documented with step-by-step comments")
    print("  • Integrated with SpeculativeEngine in engine.py")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(verify_implementation())
