#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${WOLO_VENV_DIR:-$HOME/.venvs/wolo}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found in virtual environment: $PYTHON_BIN" >&2
  echo "Run ./scripts/train.sh once (or create the venv) before starting chat." >&2
  exit 1
fi

exec "$PYTHON_BIN" src/chat.py "$@"
