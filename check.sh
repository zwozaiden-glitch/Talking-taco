#!/usr/bin/env bash
# Verify your Discord token and AI key from YOUR machine (needs internet).
#   ./check.sh            -> uses .env
#   ./check.sh .env.bot2  -> uses another config
set -uo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ -n "${1:-}" ]; then
  exec "$PY" -m talkingtaco --check --env-file "$1"
fi
exec "$PY" -m talkingtaco --check
