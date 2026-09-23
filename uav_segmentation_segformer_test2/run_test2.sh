#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
if [[ " ${*} " == *" --help "* || " ${*} " == *" --dry-run "* ]]; then
  exec python3 scripts/run_test2.py "$@"
fi
bash setup.sh
exec .venv/bin/python scripts/run_test2.py "$@"
