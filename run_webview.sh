#!/usr/bin/env bash
# Start the benchmark's browser view from the repository root.
#
#     ./run_webview.sh
#     ./run_webview.sh --port 8800 --no-browser
#
# Arguments pass through to benchmark/webview.sh, which resolves the
# interpreter (KCBENCH_PY or KCBENCH_VENV, default venv_lmm) and checks Flask
# is installed.
set -u
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark/webview.sh" "$@"
