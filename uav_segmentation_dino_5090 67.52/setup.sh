#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# A CUDA PyTorch wheel is several GiB. Keeping both the installed copy and a
# persistent pip-wheel cache can exhaust the 50GiB AutoDL data disk.
export PIP_NO_CACHE_DIR=1 TMPDIR="$ROOT/.cache/tmp" XFORMERS_DISABLED=1
mkdir -p "$TMPDIR"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys,platform; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10-3.12"; assert platform.system()=="Linux" and platform.machine()=="x86_64"'
if [[ ! -x .venv/bin/python ]]; then "$PYTHON" -m venv .venv; fi
PY="$ROOT/.venv/bin/python"
STAMP="$(sha256sum requirements.txt setup.sh | sha256sum | cut -d ' ' -f 1)"
if [[ ! -f .venv/setup.sha256 ]] || [[ "$(cat .venv/setup.sha256)" != "$STAMP" ]]; then
    "$PY" -m pip install 'pip==24.3.1'
    "$PY" -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
    "$PY" -m pip install -r requirements.txt
    "$PY" -m pip check
    printf '%s\n' "$STAMP" > .venv/setup.sha256
fi
"$PY" scripts/check_gpu.py
