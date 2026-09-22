#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 XFORMERS_DISABLED=1
if [[ ! -x .venv/bin/python ]]; then
    bash setup.sh
fi
.venv/bin/python -u predict_test2.py "$@"
