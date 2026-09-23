#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash setup.sh
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
exec .venv/bin/python pipeline.py "$@"
