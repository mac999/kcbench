#!/usr/bin/env bash
# Start the benchmark's browser view under the venv_lmm environment.
#
#     ./webview.sh
#     ./webview.sh --port 8800 --no-browser
#
# Every argument is passed straight to `cb.py webview`. Set KCBENCH_PY to point
# at a different interpreter, or KCBENCH_VENV at a different venv directory;
# the default is venv_lmm, and either a Windows (Scripts/python.exe) or a
# POSIX (bin/python) layout is found under it.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${KCBENCH_PY:-}" ]; then
    VENV="${KCBENCH_VENV:-/d/projects/adv/venv_lmm}"
    for candidate in "$VENV/Scripts/python.exe" "$VENV/bin/python"; do
        if [ -x "$candidate" ]; then KCBENCH_PY="$candidate"; break; fi
    done
fi

if [ -z "${KCBENCH_PY:-}" ] || [ ! -x "$KCBENCH_PY" ]; then
    echo "webview.sh: no interpreter found under ${KCBENCH_VENV:-/d/projects/adv/venv_lmm}" >&2
    echo "  set KCBENCH_PY to the python you want, or KCBENCH_VENV to its venv" >&2
    exit 1
fi

if ! "$KCBENCH_PY" -c "import flask" >/dev/null 2>&1; then
    echo "webview.sh: Flask is not installed in that environment. Nothing else needs it:" >&2
    echo "  \"$KCBENCH_PY\" -m pip install flask" >&2
    exit 1
fi

exec "$KCBENCH_PY" "$HERE/cb.py" webview "$@"
