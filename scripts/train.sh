#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${WOLO_VENV_DIR:-$HOME/.venvs/wolo}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
"$VENV_DIR/bin/pip" install -r requirements.txt

# Activate venv and run training
source "$VENV_DIR/bin/activate"
python3 src/main.py "$@"