#!/usr/bin/env bash
# Run any cb.py command from the repository root under the venv_lmm environment.
#
#     ./run_cli.sh                        lists the commands
#     ./run_cli.sh eval -m qwen3:8b --tag base --closed-book
#     ./run_cli.sh compare --base base --after ft-v1
#
# Arguments pass straight through. Set KCBENCH_PY to use another interpreter,
# or KCBENCH_VENV to another venv; either a Windows (Scripts/python.exe) or a
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
    echo "run_cli.sh: no interpreter found under ${KCBENCH_VENV:-/d/projects/adv/venv_lmm}" >&2
    echo "  set KCBENCH_PY to the python you want, or KCBENCH_VENV to its venv" >&2
    exit 1
fi

exec "$KCBENCH_PY" "$HERE/benchmark/cb.py" "$@"
