#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python reproduce.py "$@"
echo ""
echo "Starting Homeweave Studio & Photoreal Walkthrough Server..."
echo "  -> Constraint Studio:      http://127.0.0.1:8000"
echo "  -> Photoreal Walkthrough:  http://127.0.0.1:8000/walkthrough"
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
