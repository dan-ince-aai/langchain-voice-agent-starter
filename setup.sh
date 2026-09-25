#!/usr/bin/env bash
# Create the virtualenv and install dependencies. Requires Python 3.11+.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$OSTYPE" == darwin* ]] && ! brew list portaudio >/dev/null 2>&1; then
  echo "PortAudio is required for call.py: brew install portaudio"
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt --quiet

[ -f .env ] || cp .env.example .env

echo
echo "Set ASSEMBLYAI_API_KEY in .env, then:"
echo "    .venv/bin/python run.py"
