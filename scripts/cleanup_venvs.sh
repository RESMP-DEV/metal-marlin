#!/bin/bash
# Safe cleanup script for redundant virtual environments
# Only keeps the active .venv directory used by 'uv run'

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🔍 Identifying virtual environments..."
echo ""

# Verify active environment
ACTIVE_VENV=$(uv run python -c "import sys; print(sys.prefix)" 2>&1 | tail -1)
echo "✅ Active environment: $ACTIVE_VENV"
echo ""

# List all .venv* directories
VENV_DIRS=$(find . -maxdepth 1 -type d -name '.venv*' | sort)

if [ -z "$VENV_DIRS" ]; then
    echo "No .venv* directories found."
    exit 0
fi

echo "📦 Found virtual environments:"
du -sh .venv* 2>/dev/null | while read size dir; do
    if [[ "$PROJECT_ROOT/$dir" == "$ACTIVE_VENV" ]]; then
        echo "  $size  $dir  ← ACTIVE (will keep)"
    else
        echo "  $size  $dir  ← unused (will delete)"
    fi
done

echo ""

# Calculate total reclaimable space
TOTAL_SIZE=$(du -sh .venv_* 2>/dev/null | awk '{s+=$1} END {print s}')
if [ -n "$TOTAL_SIZE" ] && [ "$TOTAL_SIZE" != "0" ]; then
    echo "💾 Reclaimable space: ~${TOTAL_SIZE}MB"
    echo ""
fi

# Confirm deletion
read -p "⚠️  Delete unused environments? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Delete all .venv* directories except .venv
for dir in .venv_*; do
    if [ -d "$dir" ]; then
        echo "🗑️  Deleting $dir..."
        rm -rf "$dir"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining environments:"
ls -lhd .venv* 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'
