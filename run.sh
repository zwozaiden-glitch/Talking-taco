#!/usr/bin/env bash
# Start the bot. Stops with a helpful message if something is missing.
set -uo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv || exit 1
  ./.venv/bin/pip install -q -r requirements.txt || exit 1
  PY=".venv/bin/python"
fi

if [ ! -f .env ]; then
  echo "No .env found. Copying .env.example -> .env"
  cp .env.example .env
  echo "Now open .env and paste your DISCORD_TOKEN, then run ./run.sh again."
  exit 1
fi

if [ -n "${1:-}" ]; then
  exec "$PY" -m talkingtaco --env-file "$1"
fi
exec "$PY" -m talkingtaco
