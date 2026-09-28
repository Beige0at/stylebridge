#!/usr/bin/env bash
# Start the StyleBridge REST API + htmx web UI.
#   ./run_server.sh            # http://localhost:8000
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".stylebridge_env" ]; then
  echo "No .stylebridge_env found. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source ".stylebridge_env/bin/activate"

exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
