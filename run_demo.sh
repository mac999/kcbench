#!/usr/bin/env bash
# Serve the read-only demo from a bare clone on an Ubuntu host (a RunPod
# pod, a fresh VM). Handles what the plain launchers do not need to:
#
#     ./run_demo.sh            # port 8799
#     ./run_demo.sh 8888       # a RunPod base template exposes only 8888;
#                              # the Jupyter holding it is stopped first
#
# Container start command for a RunPod pod, in full:
#
#     bash -c "apt-get update -qq; apt-get install -y -qq git python3-pip;
#       git clone --depth 1 https://github.com/mac999/kcbench /app;
#       bash /app/run_demo.sh 8888"
#
# Two Ubuntu lessons are encoded here. Flask must be installed by the same
# interpreter that will run the page — on runpod/base, `pip3` belongs to
# python3.12 while `python3` is 3.10, so a bare pip3 install lands in an
# interpreter the page never sees. And Ubuntu ships a distutils blinker
# that pip cannot uninstall, which aborts a plain `pip install flask`;
# --ignore-installed steps over it.
set -u
PORT="${1:-8799}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${KCBENCH_PY:-}"
if [ -z "$PY" ]; then
    for name in python3.12 python3.11 python3 python; do
        cand="$(command -v "$name" 2>/dev/null || true)"
        if [ -n "$cand" ] && "$cand" -c "import sys" >/dev/null 2>&1; then
            PY="$cand"; break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "run_demo.sh: no python found" >&2
    exit 1
fi

if ! "$PY" -c "import flask" >/dev/null 2>&1; then
    "$PY" -m pip install --quiet --ignore-installed blinker flask || {
        echo "run_demo.sh: could not install flask into $PY" >&2; exit 1; }
fi

# If the port is already taken it is the template's Jupyter; this page is
# what the pod is for, so take the port over.
if ! "$PY" -c "import socket;s=socket.socket();s.bind(('0.0.0.0',$PORT))" \
        >/dev/null 2>&1; then
    pkill -f jupyter 2>/dev/null || true
    sleep 2
fi

exec "$PY" "$HERE/benchmark/cb.py" webview -c "$HERE/benchmark/config_kr.json" \
    --host 0.0.0.0 --port "$PORT" --no-browser --readonly
