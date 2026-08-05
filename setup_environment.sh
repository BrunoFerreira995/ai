#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    candidate_version="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${candidate_version}" == "3.11" ]]; then
      printf '%s\n' "${PYTHON_BIN}"
      return
    fi
    echo "PYTHON_BIN must point to Python 3.11, found ${candidate_version}." >&2
    exit 1
  fi

  for candidate in python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      candidate_version="$(${candidate} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "${candidate_version}" == "3.11" ]]; then
        printf '%s\n' "${candidate}"
        return
      fi
    fi
  done

  if [[ -x /opt/homebrew/bin/python3.11 ]]; then
    printf '%s\n' /opt/homebrew/bin/python3.11
    return
  fi

  if [[ -x /usr/local/bin/python3.11 ]]; then
    printf '%s\n' /usr/local/bin/python3.11
    return
  fi

  echo "Python 3.11 was not found. Install it or set PYTHON_BIN=/path/to/python3.11." >&2
  exit 1
}

PYTHON="$(find_python)"
PYTHON_VERSION="$(${PYTHON} --version 2>&1)"

echo "Using ${PYTHON} (${PYTHON_VERSION})"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Missing requirements file: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_VERSION="$(${VENV_DIR}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${VENV_VERSION}" != "3.11" ]]; then
    echo "Replacing existing virtual environment (Python ${VENV_VERSION})."
    mv "${VENV_DIR}" "${PROJECT_DIR}/.venv-py${VENV_VERSION//./}"
  fi
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating ${VENV_DIR}..."
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

echo "Upgrading pip..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip

echo "Installing project libraries..."
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"

echo "Checking dependency consistency..."
"${VENV_DIR}/bin/python" -m pip check

echo "Validating installed libraries..."
"${VENV_DIR}/bin/python" - <<'PY'
import importlib

modules = {
    "TensorFlow": "tensorflow",
    "Keras": "keras",
    "NumPy": "numpy",
    "Pandas": "pandas",
    "Scikit-learn": "sklearn",
    "Matplotlib": "matplotlib",
    "OpenCV": "cv2",
    "TensorBoard": "tensorboard",
    "TensorFlow Datasets": "tensorflow_datasets",
    "TensorFlow Hub": "tensorflow_hub",
    "TensorFlow Addons": "tensorflow_addons",
    "Albumentations": "albumentations",
    "ONNX": "onnx",
    "tf2onnx": "tf2onnx",
}

failed = []
for package, module in modules.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "")
        print(f"  OK  {package}{f' ({version})' if version else ''}")
    except Exception as error:
        failed.append((package, error))
        print(f"  FAIL {package}: {error}")

if failed:
    raise SystemExit(1)
PY

"${VENV_DIR}/bin/python" -m pip check

echo
echo "Environment setup completed successfully."
echo "Activate it with: source .venv/bin/activate"
