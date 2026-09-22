#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
trap 'code=$?; echo "Failed ($code). Retain runs/ and weights. Resume with the same config, without --new-run."; exit "$code"' ERR
bash setup.sh
PY="${DINOV3_ENV:-$ROOT/.venv}/bin/python"
"$PY" scripts/local_checks.py
"$PY" -m unittest discover -s tests -v
"$PY" -u pipeline.py "$@"
