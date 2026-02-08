#!/bin/bash
# Script to run performance tracking for Metal Marlin

set -e

echo "📊 Running continuous performance tracking..."

cd "$(dirname "$0")/.."

# Check if MPS is available
if ! uv run python -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
    echo "❌ MPS not available. Cannot run performance benchmarks."
    echo "   Ensure you're on Apple Silicon with PyTorch MPS backend."
    exit 1
fi

# Run the performance tracking
echo "Running kernel and E2E benchmarks..."
uv run python benchmarks/continuous_perf_tracking.py \
    --output benchmarks/results/latest_perf.json \
    --report benchmarks/results/latest_perf.md \
    --fail-on-regression \
    --ci-threshold 0.10

# Check if baseline exists
BASELINE="benchmarks/baseline_perf.json"
if [ ! -f "$BASELINE" ]; then
    echo "⚠️  No baseline found. Creating initial baseline..."
    uv run python benchmarks/continuous_perf_tracking.py --update-baseline
fi

echo "✅ Performance tracking complete!"
echo "Results saved to benchmarks/results/latest_perf.json"
echo "Report saved to benchmarks/results/latest_perf.md"
