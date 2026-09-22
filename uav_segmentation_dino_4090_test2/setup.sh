#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp" XFORMERS_DISABLED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys,platform; assert sys.version_info[:2]==(3,10), "Use Python 3.10"; assert platform.system()=="Linux" and platform.machine()=="x86_64"'
if [[ ! -x .venv/bin/python ]]; then "$PYTHON" -m venv .venv; fi
PY="$ROOT/.venv/bin/python"
STAMP="$(sha256sum requirements.txt setup.sh | sha256sum | cut -d ' ' -f 1)"
if [[ ! -f .venv/setup.sha256 ]] || [[ "$(cat .venv/setup.sha256)" != "$STAMP" ]]; then
    "$PY" -m pip install 'pip==24.3.1'
    "$PY" -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
    "$PY" -m pip install -r requirements.txt
    "$PY" -m pip check
    printf '%s\n' "$STAMP" > .venv/setup.sha256
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__,torch.cuda.get_device_name())'
