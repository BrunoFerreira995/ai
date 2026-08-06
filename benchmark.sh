#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
MODEL="${MODEL:-${PROJECT_DIR}/artifacts/saved_model}"
BENCHMARKS="${BENCHMARKS:-mmlu,mmlu_pro,gpqa_diamond,math,ifeval}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Ambiente não encontrado. Execute ./install.sh primeiro." >&2
  exit 1
fi

exec "${PYTHON}" "${PROJECT_DIR}/benchmarks/run_benchmarks.py" \
  --model "${MODEL}" \
  --benchmark "${BENCHMARKS}" \
  "$@"
