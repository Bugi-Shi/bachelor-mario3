#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BIN="$ROOT_DIR/external/gym-retro/gym-retro-integration"
ALT_BIN="$ROOT_DIR/external/gym-retro/build/gym-retro-integration"

if [[ ! -x "$BIN" ]]; then
  if [[ -x "$ALT_BIN" ]]; then
    BIN="$ALT_BIN"
  else
    echo "Binary nicht gefunden: $BIN" >&2
    echo "Baue es zuerst mit: bash tools/retro_integration/build.sh" >&2
    exit 1
  fi
fi

"$BIN"
