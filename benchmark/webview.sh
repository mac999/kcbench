#!/usr/bin/env bash
# Start the benchmark's browser view.
#
#     ./webview.sh
#     ./webview.sh --port 8800 --no-browser
#
# Every argument is passed straight to `cb.py webview`. The interpreter is
# KCBENCH_PY if set, else a venv/ or .venv/ in the repository root (Windows or
# POSIX layout), else the python on PATH. KCBENCH_VENV names a venv anywhere
# else. Flask is checked before starting, since nothing else needs it.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

if [ -z "${KCBENCH_PY:-}" ]; then
    for candidate in \
        ${KCBENCH_VENV:+"$KCBENCH_VENV/Scripts/python.exe" "$KCBENCH_VENV/bin/python"} \
        "$ROOT/venv/Scripts/python.exe"  "$ROOT/venv/bin/python" \
        "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python"; do
        if [ -x "$candidate" ]; then KCBENCH_PY="$candidate"; break; fi
    done
fi
if [ -z "${KCBENCH_PY:-}" ]; then
    # the Windows Store ships a python3 stub that only prints an install hint,
    # so a candidate counts only if it actually runs
    for name in python3 python; do
        cand="$(command -v "$name" 2>/dev/null || true)"
        if [ -n "$cand" ] && "$cand" -c "import sys" >/dev/null 2>&1; then
            KCBENCH_PY="$cand"; break
        fi
    done
fi

if [ -z "${KCBENCH_PY:-}" ] || [ ! -x "$KCBENCH_PY" ]; then
    echo "webview.sh: no python found. Make a venv in the repository root" >&2
    echo "  (python -m venv venv) or set KCBENCH_PY / KCBENCH_VENV." >&2
    exit 1
fi

if ! "$KCBENCH_PY" -c "import flask" >/dev/null 2>&1; then
    echo "webview.sh: Flask is not installed in that environment. Nothing else needs it:" >&2
    echo "  \"$KCBENCH_PY\" -m pip install flask" >&2
    exit 1
fi

exec "$KCBENCH_PY" "$HERE/cb.py" webview "$@"
