#!/usr/bin/env bash
# StyleBridge one-shot environment setup.
#   ./setup.sh
# Creates a virtualenv at .stylebridge_env/ and installs requirements.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".stylebridge_env"

echo "[setup] creating virtualenv in $VENV (python: $PY)"
"$PY" -m venv "$VENV"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip

echo "[setup] installing requirements"
pip install -r requirements.txt

echo
echo "[setup] done."
echo "Activate the environment with:  source $VENV/bin/activate"
echo "Then start the server with:     uvicorn api:app --port 8000"
