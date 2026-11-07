#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .venv/Scripts/python.exe ]]; then
  python_path=./.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then
  python_path=./.venv/bin/python
else
  printf '%s\n' 'Create the Python environment first: uv sync --locked --python 3.13' >&2
  exit 1
fi

export VG_REQUIRE_ALL_MODELS=1
exec "$python_path" -m uvicorn visionguard.api.main:app --host 127.0.0.1 --port 8000
