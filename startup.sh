#!/bin/bash
# startup.sh — Setup and launch the FastAPI server
#
# Usage:
#   ./startup.sh              # default: uses .venv, port 7860
#   PORT=8080 ./startup.sh    # custom port
#   ./startup.sh --install    # install deps first

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
VENV_DIR="${VENV_DIR:-.venv}"

# ------------------------------------------------------------------
# Install dependencies (optional flag)
# ------------------------------------------------------------------
if [[ "$1" == "--install" ]]; then
    echo "[startup] Installing dependencies..."
    if [ -f "$VENV_DIR/bin/pip" ]; then
        "$VENV_DIR/bin/pip" install -q -r requirements.txt
        "$VENV_DIR/bin/pip" install -q fastapi uvicorn python-multipart
    else
        echo "[startup] Virtual env not found at $VENV_DIR — please create it first:"
        echo "  python -m venv $VENV_DIR"
        echo "  source $VENV_DIR/bin/activate"
        echo "  pip install -r requirements.txt fastapi uvicorn python-multipart"
        exit 1
    fi
    echo "[startup] Dependencies installed."
fi

# ------------------------------------------------------------------
# Check virtual environment
# ------------------------------------------------------------------
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[startup] ERROR: Virtual environment not found at $VENV_DIR"
    echo "[startup] Run: python -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -r requirements.txt fastapi uvicorn python-multipart"
    exit 1
fi

# ------------------------------------------------------------------
# Check for required env vars (warn but don't fail)
# ------------------------------------------------------------------
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "[startup] WARNING: OPENROUTER_API_KEY not set — LLM fusion will fall back to heuristic"
fi

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
echo "[startup] Starting FastAPI server on http://$HOST:$PORT"
echo "[startup] Press Ctrl+C to stop"
"$VENV_DIR/bin/uvicorn" fastapi_app:app --host "$HOST" --port "$PORT" --reload
