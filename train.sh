#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Ambiente não encontrado. Execute ./install.sh primeiro." >&2
  exit 1
fi

exec "${PYTHON}" "${PROJECT_DIR}/train.py" "$@"
