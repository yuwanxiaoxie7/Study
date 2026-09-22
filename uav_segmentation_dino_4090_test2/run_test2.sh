#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo '复赛：按4090初赛配方重新训练30轮，自动选择最佳权重并预测测试集2。'
exec bash run_all.sh "$@"
