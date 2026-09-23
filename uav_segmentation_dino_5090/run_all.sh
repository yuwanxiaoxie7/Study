#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 XFORMERS_DISABLED=1
if [[ $# -eq 0 ]] && [[ -z "${AIC_DATA_ROOT:-}" ]]; then
    echo 'Usage: bash run_all.sh --data-root /path/to/dataset [--check-only]'
    exit 2
fi
trap 'result=$?; echo "Failed ($result). See runs/<experiment>/console.log; rerun the same command to resume."; exit "$result"' ERR
"${PYTHON:-python3}" scripts/preflight.py "$@"
bash setup.sh
.venv/bin/python scripts/local_checks.py
.venv/bin/python scripts/verify_sources.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u pipeline.py "$@"
