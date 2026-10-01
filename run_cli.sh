#!/usr/bin/env bash
# Run any cb.py command from the repository root.
#
#     ./run_cli.sh                        lists the commands
#     ./run_cli.sh eval -m qwen3:8b --tag base --closed-book
#     ./run_cli.sh compare --base base --after ft-v1
#
# Arguments pass straight through. The interpreter is KCBENCH_PY if set, else
# a venv/ or .venv/ inside the repository (Windows or POSIX layout), else the
# python on PATH. KCBENCH_VENV names a venv anywhere else.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${KCBENCH_PY:-}" ]; then
    for candidate in \
        ${KCBENCH_VENV:+"$KCBENCH_VENV/Scripts/python.exe" "$KCBENCH_VENV/bin/python"} \
        "$HERE/venv/Scripts/python.exe"  "$HERE/venv/bin/python" \
        "$HERE/.venv/Scripts/python.exe" "$HERE/.venv/bin/python"; do
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
    echo "run_cli.sh: no python found. Make a venv in the repository root" >&2
    echo "  (python -m venv venv) or set KCBENCH_PY / KCBENCH_VENV." >&2
    exit 1
fi

exec "$KCBENCH_PY" "$HERE/benchmark/cb.py" "$@"
