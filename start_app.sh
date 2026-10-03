#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Virtual environment not found at .venv" >&2
  exit 1
fi

source .venv/bin/activate

PORT="${STREAMLIT_SERVER_PORT:-8502}"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_PORT="$PORT"

printf 'Starting Streamlit on http://localhost:%s\n' "$PORT"
exec streamlit run streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port "$PORT"
