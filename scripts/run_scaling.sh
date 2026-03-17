#!/usr/bin/env bash
# scripts/run_scaling.sh — Run scaling experiments v1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

CONFIG="${1:-configs/experiments/scaling_v1.yaml}"
DRY_RUN="${DRY_RUN:-false}"

echo "=== Scaling Experiment Runner ==="
echo "Config: $CONFIG"

if [ "$DRY_RUN" = "true" ]; then
    echo "--- DRY RUN: printing config and exiting ---"
    python -m iqp_bp.cli run-scaling "$CONFIG" --dry-run
    exit 0
fi

echo "Starting run at $(date)..."
python -m iqp_bp.cli run-scaling "$CONFIG"
echo "Done at $(date)."
echo "Results: $(grep -o '"output_dir":[^,]*' "$CONFIG" | head -1 || echo 'see config')"
