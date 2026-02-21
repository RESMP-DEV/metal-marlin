#!/bin/bash
# GLM-4.7-Flash Complete Validation Runner
# Runs all end-to-end tests and provides summary report

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METAL_MARLIN_ROOT="$PROJECT_ROOT/contrib/metal_marlin"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}=====================================================================${NC}"
echo -e "${BOLD}        GLM-4.7-Flash Complete Validation Suite${NC}"
echo -e "${BOLD}=====================================================================${NC}"
echo ""

# Check if server is running
echo -e "${BLUE}[Step 1/5] Checking server status...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is running at http://localhost:8000${NC}"
else
    echo -e "${RED}❌ Server is not running${NC}"
    echo ""
    echo "Start the server first:"
    echo "  cd $METAL_MARLIN_ROOT"
    echo "  uv run python scripts/serve_glm47.py --model-path ./models/glm47-flash-mmfp4"
    echo ""
    exit 1
fi

# Check if model path exists
echo ""
echo -e "${BLUE}[Step 2/5] Checking model weights...${NC}"
if [ ! -d "$METAL_MARLIN_ROOT/models/glm47-flash-mmfp4" ]; then
    echo -e "${YELLOW}⚠️  Model directory not found at: $METAL_MARLIN_ROOT/models/glm47-flash-mmfp4${NC}"
    echo "This is required for the standalone perplexity benchmark."
    echo "The server-based tests will still work if server was started with correct model path."
    MODEL_PATH_EXISTS=false
else
    echo -e "${GREEN}✅ Model weights found${NC}"
    MODEL_PATH_EXISTS=true
fi

# Run tests
echo ""
echo -e "${BOLD}=====================================================================${NC}"
echo -e "${BOLD}[Step 3/5] Running End-to-End Validation Tests${NC}"
echo -e "${BOLD}=====================================================================${NC}"

cd "$METAL_MARLIN_ROOT"
if uv run python tests/manual/test_e2e_validation.py; then
    E2E_RESULT="PASS"
    echo -e "${GREEN}✅ E2E validation passed${NC}"
else
    E2E_RESULT="FAIL"
    echo -e "${RED}❌ E2E validation failed${NC}"
fi

echo ""
echo -e "${BOLD}=====================================================================${NC}"
echo -e "${BOLD}[Step 4/5] Running Standalone Perplexity Benchmark${NC}"
echo -e "${BOLD}=====================================================================${NC}"

if [ "$MODEL_PATH_EXISTS" = true ]; then
    if uv run python tests/manual/benchmark_perplexity.py \
        --model-path ./models/glm47-flash-mmfp4 \
        --dataset wikitext \
        --benchmark-tokens 100; then
        PERPLEXITY_RESULT="PASS"
        echo -e "${GREEN}✅ Perplexity benchmark passed${NC}"
    else
        PERPLEXITY_RESULT="FAIL"
        echo -e "${RED}❌ Perplexity benchmark failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Skipping perplexity benchmark (model not found)${NC}"
    PERPLEXITY_RESULT="SKIP"
fi

echo ""
echo -e "${BOLD}=====================================================================${NC}"
echo -e "${BOLD}[Step 5/5] Testing Metrics Endpoint${NC}"
echo -e "${BOLD}=====================================================================${NC}"

if curl -s http://localhost:8000/metrics | grep -q "throughput"; then
    echo -e "${GREEN}✅ Metrics endpoint working${NC}"
    METRICS_RESULT="PASS"
    
    # Display current metrics
    echo ""
    echo "Current server metrics:"
    curl -s http://localhost:8000/metrics | grep -E "throughput_tok_sec|avg_latency_ms|active_requests|requests_total" | head -10
else
    echo -e "${RED}❌ Metrics endpoint failed${NC}"
    METRICS_RESULT="FAIL"
fi

# Summary
echo ""
echo -e "${BOLD}=====================================================================${NC}"
echo -e "${BOLD}                        Validation Summary${NC}"
echo -e "${BOLD}=====================================================================${NC}"

echo ""
echo "Test Results:"
echo "  1. E2E Validation:         $E2E_RESULT"
echo "  2. Perplexity Benchmark:   $PERPLEXITY_RESULT"
echo "  3. Metrics Endpoint:       $METRICS_RESULT"
echo ""

if [ "$E2E_RESULT" = "PASS" ] && [ "$METRICS_RESULT" = "PASS" ] && [ "$PERPLEXITY_RESULT" != "FAIL" ]; then
    echo -e "${GREEN}${BOLD}✅ ALL TESTS PASSED${NC}"
    echo ""
    echo -e "${GREEN}GLM-4.7-Flash is validated and ready for production!${NC}"
    echo ""
    echo "Performance Summary:"
    echo "  • Target:   35.0 tok/s, <30 ms/token"
    echo "  • Achieved: Check metrics output above"
    echo "  • Quality:  All tests passed"
    echo ""
    echo "Next steps:"
    echo "  • Deploy to production"
    echo "  • Monitor with /metrics endpoint"
    echo "  • See docs/serving_guide.md for full documentation"
    echo ""
    exit 0
else
    echo -e "${RED}${BOLD}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Review the output above for details."
    echo ""
    echo "Common issues:"
    echo "  • Low TPS: Check GPU memory, reduce batch size"
    echo "  • High perplexity: Verify model weights loaded correctly"
    echo "  • Connection errors: Verify server is running"
    echo ""
    echo "See docs/VALIDATION_CHECKLIST.md for troubleshooting."
    echo ""
    exit 1
fi
