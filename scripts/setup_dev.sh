#!/usr/bin/env bash
# scripts/setup_dev.sh — Set up development environment
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== IQP–MMD Barren Plateau: Dev Setup ==="
echo "Working dir: $ROOT"

# Create and activate venv
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python -m venv .venv
else
    echo "[1/4] Virtual environment already exists."
fi

# Activate
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "[2/4] Upgrading pip..."
pip install -U pip

echo "[3/4] Installing core dependencies..."
pip install -r requirements.txt

echo "[4/4] Installing package in editable mode..."
pip install -e ".[dev]"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: source .venv/Scripts/activate  (Windows)"
echo "               source .venv/bin/activate       (Linux/macOS)"
echo ""
echo "Quick test: python -m pytest tests/ -x -q"
