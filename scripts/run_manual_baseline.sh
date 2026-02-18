#!/bin/bash
# Manual Baseline Benchmark Runner
#
# CRITICAL: Run this MANUALLY, NOT as an AlphaHENG task.
# Benchmarks load large models and can spike to 120GB+ memory.
#
# Usage:
#   cd contrib/metal_marlin
#   ./scripts/run_manual_baseline.sh [--quick] [--skip-heavy]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"

# Ensure we're NOT in task mode
if [ "${ALPHAHENG_TASK_MODE:-0}" = "1" ]; then
    echo "ERROR: This script cannot run in AlphaHENG task mode."
    echo "Run it manually in a separate terminal."
    exit 1
fi

# Parse arguments
QUICK=0
SKIP_HEAVY=0
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=1
            shift
            ;;
        --skip-heavy)
            SKIP_HEAVY=1
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--quick] [--skip-heavy]"
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo "GLM-4.7-Flash Manual Baseline Benchmark"
echo "========================================================================"
echo ""
echo "This will run benchmarks OUTSIDE of AlphaHENG task mode."
echo "Expected memory usage: 10-20 GB per benchmark"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Check if quantized model exists and is compatible
if [ ! -d "$PROJECT_ROOT/models/GLM-4.7-Flash-Trellis-MM" ]; then
    echo "⚠️  WARNING: Quantized model not found at models/GLM-4.7-Flash-Trellis-MM"
    echo "⚠️  Will skip quantized model baseline."
    echo ""
fi

# Change to project directory
cd "$PROJECT_ROOT"

# ========================================================================
# 1. Quick Baseline (lightweight, ~10 GB)
# ========================================================================
echo ""
echo "--------------------------------------------------------------------"
echo "[1/4] Running quick baseline..."
echo "--------------------------------------------------------------------"
uv run python benchmarks/glm47_baseline.py \
    --quick \
    --skip-perplexity \
    --output "$RESULTS_DIR/glm47_baseline_quick.json"

if [ $? -ne 0 ]; then
    echo "ERROR: Quick baseline failed"
    exit 1
fi

echo "✅ Quick baseline complete"

# ========================================================================
# 2. Full Baseline (heavier, ~15 GB, includes perplexity)
# ========================================================================
if [ $SKIP_HEAVY -eq 0 ]; then
    echo ""
    echo "--------------------------------------------------------------------"
    echo "[2/4] Running full baseline with perplexity..."
    echo "--------------------------------------------------------------------"
    echo "⚠️  This will take 5-10 minutes and use ~15 GB memory"
    
    uv run python benchmarks/glm47_baseline.py \
        --ppl-samples 50 \
        --context-length 512 \
        --decode-tokens 100 \
        --output "$RESULTS_DIR/glm47_baseline_full.json"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Full baseline failed"
        exit 1
    fi
    
    echo "✅ Full baseline complete"
else
    echo ""
    echo "[2/4] Skipping full baseline (--skip-heavy)"
fi

# ========================================================================
# 3. Mixed BPW Distribution Analysis (lightweight)
# ========================================================================
echo ""
echo "--------------------------------------------------------------------"
echo "[3/4] Analyzing bit-per-weight distribution..."
echo "--------------------------------------------------------------------"

if [ -f "benchmarks/analyze_bpw_distribution.py" ]; then
    uv run python benchmarks/analyze_bpw_distribution.py \
        --model models/GLM-4.7-Flash-Trellis-MM \
        --output "$RESULTS_DIR/bpw_distribution.json" || true
    echo "✅ BPW analysis complete"
else
    echo "⚠️  benchmarks/analyze_bpw_distribution.py not found (optional)"
fi

# ========================================================================
# 4. Generate Summary
# ========================================================================
echo ""
echo "--------------------------------------------------------------------"
echo "[4/4] Generating summary report..."
echo "--------------------------------------------------------------------"

SUMMARY="$RESULTS_DIR/BASELINE_SUMMARY.txt"
cat > "$SUMMARY" <<EOF
GLM-4.7-Flash Baseline Summary
Generated: $(date)
======================================================================

Results Location: $RESULTS_DIR

EOF

# Extract key metrics from quick baseline
if [ -f "$RESULTS_DIR/glm47_baseline_quick.json" ]; then
    echo "Quick Baseline Results:" >> "$SUMMARY"
    jq -r '
        "  Decode: \(.throughput.decode_tok_s | tonumber | . * 10 | round / 10) tok/s (\(.throughput.decode_ms_per_token | tonumber | . * 10 | round / 10) ms/tok)",
        "  Prefill 128: \(.throughput.prefill_128_tok_s | tonumber | . * 10 | round / 10) tok/s",
        "  Memory: \(.memory.allocated_gb | tonumber | . * 100 | round / 100) GB"
    ' "$RESULTS_DIR/glm47_baseline_quick.json" >> "$SUMMARY" 2>/dev/null || echo "  (parsing error)" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
fi

# Extract perplexity if available
if [ -f "$RESULTS_DIR/glm47_baseline_full.json" ]; then
    echo "Full Baseline Results:" >> "$SUMMARY"
    jq -r '
        if .perplexity then
            "  WikiText-2 PPL: \(.perplexity.ppl | tonumber | . * 100 | round / 100)"
        else
            "  (no perplexity data)"
        end
    ' "$RESULTS_DIR/glm47_baseline_full.json" >> "$SUMMARY" 2>/dev/null || echo "  (parsing error)" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
fi

echo "======================================================================" >> "$SUMMARY"
echo "Next Steps:" >> "$SUMMARY"
echo "1. Review results in $RESULTS_DIR" >> "$SUMMARY"
echo "2. Identify bottlenecks from the metrics" >> "$SUMMARY"
echo "3. Create optimization tasks targeting specific bottlenecks" >> "$SUMMARY"
echo "4. Update README.md with actual baseline numbers" >> "$SUMMARY"
echo "======================================================================" >> "$SUMMARY"

cat "$SUMMARY"

echo ""
echo "========================================================================"
echo "✅ Manual baseline benchmarks complete!"
echo "========================================================================"
echo ""
echo "Results saved to:  $RESULTS_DIR"
echo "Summary saved to:  $SUMMARY"
echo ""
echo "Next: Review the numbers and create targeted optimization tasks."
echo ""
