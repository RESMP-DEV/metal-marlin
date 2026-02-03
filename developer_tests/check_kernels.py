import sys
from io import StringIO

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

# Capture stderr to detect Metal compilation errors
old_stderr = sys.stderr
sys.stderr = captured = StringIO()

try:
    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

    print("Testing forward pass...")
    x = torch.randint(0, 1000, (1, 64)).to('mps')
    with torch.no_grad():
        out = model(x)

    print("Forward pass completed")
finally:
    sys.stderr = old_stderr
    errors = captured.getvalue()

# Parse Metal errors
metal_errors = []
for line in errors.split('\n'):
    if 'error:' in line.lower() or 'program_source' in line:
        metal_errors.append(line)

if metal_errors:
    print(f"\nFound {len(metal_errors)} Metal compilation errors:")
    for err in metal_errors[:10]:
        print(f"  {err}")
else:
    print("\n✓ No Metal compilation errors detected")
