#!/usr/bin/env bash
# One-time setup. Python 3.11 or newer.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$OSTYPE" == darwin* ]] && ! brew list portaudio >/dev/null 2>&1; then
  echo "call.py needs PortAudio for the microphone:  brew install portaudio"
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt --quiet

[ -f .env ] || cp .env.example .env

echo
echo "Done. Put your key in .env, then:"
echo "    .venv/bin/python run.py"
