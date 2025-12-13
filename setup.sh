#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for this repo.
# Goal: after `git pull` and installing Python 3.8, run `./setup.sh` and you're ready.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-project}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "ERROR: '$PYTHON_BIN' not found." >&2
	echo "Install Python 3.8 first, then re-run: ./setup.sh" >&2
	echo "Tip (Ubuntu/Debian): sudo apt-get install python3.8 python3.8-venv" >&2
	exit 1
fi

echo ">>> Using Python: $($PYTHON_BIN --version 2>&1)"

# Ensure venv module exists
if ! "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
	echo "ERROR: Python venv module not available for '$PYTHON_BIN'." >&2
	echo "On Ubuntu/Debian, install: sudo apt-get install python3.8-venv" >&2
	exit 1
fi

if [[ ! -d "$VENV_DIR" || ! -x "$VENV_DIR/bin/python" ]]; then
	echo ">>> Creating virtual environment in: $VENV_DIR"
	"$PYTHON_BIN" -m venv "$VENV_DIR"
else
	echo ">>> Reusing existing virtual environment: $VENV_DIR"
fi

VENV_PY="$ROOT_DIR/$VENV_DIR/bin/python"

echo ">>> Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip

REQ_LOCK_FILE="${REQ_LOCK_FILE:-requirements-lock.txt}"
REQ_DEV_FILE="${REQ_DEV_FILE:-requirements-dev.txt}"

if [[ -f "$REQ_LOCK_FILE" ]]; then
	echo ">>> Installing dependencies ($REQ_LOCK_FILE)"
	"$VENV_PY" -m pip install -r "$REQ_LOCK_FILE"
else
	echo "ERROR: $REQ_LOCK_FILE not found in repo root." >&2
	echo "If your file has a different name, run e.g.:" >&2
	echo "  REQ_LOCK_FILE=<yourfile>.txt ./setup.sh" >&2
	exit 1
fi

if [[ -f "$REQ_DEV_FILE" ]]; then
	echo ">>> Installing dev dependencies ($REQ_DEV_FILE)"
	"$VENV_PY" -m pip install -r "$REQ_DEV_FILE"
fi

# Optional: if a local stable-retro checkout exists, install it editable.
# This can be useful when you need the repo's tooling/data layout.
if [[ -d "stable-retro" ]]; then
	echo ">>> Found ./stable-retro (installing editable)"
	"$VENV_PY" -m pip install -e "./stable-retro"
fi

echo ">>> Done. Activate with: source $VENV_DIR/bin/activate"
