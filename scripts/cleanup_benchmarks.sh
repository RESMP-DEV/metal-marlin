#!/usr/bin/env bash
# Cleanup temporary files and logs from benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Cleaning benchmark artifacts in $PROJECT_ROOT"

# Remove Python cache files
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove benchmark logs
rm -rf "$PROJECT_ROOT/logs"/*.log 2>/dev/null || true

# Remove benchmark reports
rm -rf "$PROJECT_ROOT/reports"/*.json 2>/dev/null || true
rm -rf "$PROJECT_ROOT/reports"/*.html 2>/dev/null || true
rm -rf "$PROJECT_ROOT/reports"/*.csv 2>/dev/null || true

# Remove benchmark results
rm -rf "$PROJECT_ROOT/results"/*.json 2>/dev/null || true
rm -rf "$PROJECT_ROOT/results"/*.txt 2>/dev/null || true

# Remove agent workspace directories
rm -rf "$PROJECT_ROOT/agent_workspace"/opt_* 2>/dev/null || true
rm -rf "$PROJECT_ROOT/agent_workspace"/struct_* 2>/dev/null || true
rm -rf "$PROJECT_ROOT/agent_workspace"/*.yaml 2>/dev/null || true

# Remove temporary files
find "$PROJECT_ROOT" -type f -name "*.tmp" -delete 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.temp" -delete 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name ".DS_Store" -delete 2>/dev/null || true

# Remove benchmark-specific temp files
rm -f "$PROJECT_ROOT/benchmarks"/.DS_Store 2>/dev/null || true
rm -rf "$PROJECT_ROOT/benchmarks/__pycache__" 2>/dev/null || true

echo "✓ Cleanup complete"
