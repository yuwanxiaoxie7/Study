#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TMPDIR="$ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys,platform; assert (3,10)<=sys.version_info[:2]<=(3,12); assert platform.system()=="Linux" and platform.machine()=="x86_64"'
"$PYTHON" -c 'import shutil; free=shutil.disk_usage(".").free/2**30; print("Disk free GiB:",round(free,1)); assert free>=8, "Free at least 8 GiB before setup; no files deleted"'
ENV_DIR="${DINOV3_ENV:-$ROOT/.venv}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$PYTHON" -m venv "$ENV_DIR"; fi
PY="$ENV_DIR/bin/python"
# Reuse an explicitly supplied compatible environment. Never delete or recreate it.
if ! "$PY" -c 'import torch; assert torch.__version__.split("+")[0]=="2.7.1" and torch.version.cuda=="12.8"' 2>/dev/null; then
    "$PY" -m pip install 'torch==2.7.1' --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
fi
"$PY" -m pip install -r requirements.txt --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}"
"$PY" -m pip check
"$PY" -c 'import torch; print("Environment ready:",torch.__version__,"CUDA runtime",torch.version.cuda,"GPU available",torch.cuda.is_available())'
echo 'CPU-only preparation is supported. GPU kernels are checked by run_all.sh before benchmarking/training.'
